"""
Fact Prediction Script
======================
Load a trained KGE model (TransE, ComplEx, or TriModel) and predict scores
for specific facts (triples).

Usage:
    python predict_fact.py
    
The script will:
1. Load a trained model checkpoint
2. Load entity/relation mappings
3. Score user-specified facts
4. Optionally predict top-k tail/head entities for a given query
"""

import os
import torch
import numpy as np
import pandas as pd

from model import TransE, ComplEx, TriModel


# ----------------------------
# Load trained models
# ----------------------------
def load_transe_model(checkpoint_path: str, device: str = "cpu"):
    """Load a trained TransE model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    model = TransE(
        num_entities=ckpt["num_entities"],
        num_relations=ckpt["num_relations"],
        dim=ckpt["embedding_dim"],
        p_norm=ckpt["p_norm"],
    ).to(device)
    
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    return model, ckpt["entity2id"], ckpt["relation2id"]


def load_complex_model(checkpoint_path: str, device: str = "cpu"):
    """Load a trained ComplEx model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    model = ComplEx(
        num_entities=ckpt["num_entities"],
        num_relations=ckpt["num_relations"],
        dim=ckpt["embedding_dim"],
        reg_weight=ckpt.get("reg_weight", 0.01),
    ).to(device)
    
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    return model, ckpt["entity2id"], ckpt["relation2id"]


def load_trimodel_model(checkpoint_path: str, device: str = "cpu"):
    """Load a trained TriModel from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    model = TriModel(
        num_entities=ckpt["num_entities"],
        num_relations=ckpt["num_relations"],
        dim=ckpt["embedding_dim"],
        reg_weight=ckpt.get("reg_weight", 0.01),
    ).to(device)
    
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    return model, ckpt["entity2id"], ckpt["relation2id"]


def load_model(checkpoint_path: str, device: str = "cpu"):
    """
    Auto-detect and load any supported model type.
    
    Returns:
        model: Loaded model
        entity2id: Entity to ID mapping
        relation2id: Relation to ID mapping
        model_type: String indicating model type
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Detect model type from checkpoint
    model_type = ckpt.get("model_type", None)
    
    if model_type is None:
        # Infer from checkpoint keys
        if "p_norm" in ckpt:
            model_type = "TransE"
        elif "reg_weight" in ckpt:
            # Could be ComplEx or TriModel - check for specific keys
            model_type = "ComplEx"  # Default to ComplEx
    
    print(f"Detected model type: {model_type}")
    
    if model_type == "TransE":
        model, entity2id, relation2id = load_transe_model(checkpoint_path, device)
    elif model_type == "ComplEx":
        model, entity2id, relation2id = load_complex_model(checkpoint_path, device)
    elif model_type == "TriModel":
        model, entity2id, relation2id = load_trimodel_model(checkpoint_path, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, entity2id, relation2id, model_type


# ----------------------------
# Prediction functions
# ----------------------------
@torch.no_grad()
def score_facts(
    model, 
    facts: list, 
    entity2id: dict, 
    relation2id: dict, 
    model_type: str,
    device: str = "cpu"
) -> list:
    """
    Score a list of facts (triples).
    
    Args:
        model: Trained KGE model
        facts: List of [head, relation, tail] triples (using original names)
        entity2id: Entity to ID mapping
        relation2id: Relation to ID mapping
        model_type: Type of model ('TransE', 'ComplEx', 'TriModel')
        device: Device to run on
        
    Returns:
        List of (fact, score, is_valid) tuples
    """
    results = []
    
    for fact in facts:
        head, relation, tail = fact
        
        # Check if all components are in the vocabulary
        if head not in entity2id:
            results.append((fact, None, f"Unknown head entity: {head}"))
            continue
        if relation not in relation2id:
            results.append((fact, None, f"Unknown relation: {relation}"))
            continue
        if tail not in entity2id:
            results.append((fact, None, f"Unknown tail entity: {tail}"))
            continue
        
        # Encode the triple
        h_id = entity2id[head]
        r_id = relation2id[relation]
        t_id = entity2id[tail]
        
        triple_tensor = torch.tensor([[h_id, r_id, t_id]], dtype=torch.long, device=device)
        
        # Get score from model
        score = model(triple_tensor).item()
        
        results.append((fact, score, "OK"))
    
    return results


@torch.no_grad()
def predict_tails(
    model,
    head: str,
    relation: str,
    entity2id: dict,
    relation2id: dict,
    model_type: str,
    top_k: int = 10,
    device: str = "cpu"
) -> list:
    """
    Predict top-k tail entities for a given (head, relation, ?).
    
    Args:
        model: Trained KGE model
        head: Head entity name
        relation: Relation name
        entity2id: Entity to ID mapping
        relation2id: Relation to ID mapping
        model_type: Type of model
        top_k: Number of top predictions to return
        device: Device to run on
        
    Returns:
        List of (entity, score) tuples sorted by score
    """
    if head not in entity2id:
        raise ValueError(f"Unknown head entity: {head}")
    if relation not in relation2id:
        raise ValueError(f"Unknown relation: {relation}")
    
    h_id = entity2id[head]
    r_id = relation2id[relation]
    
    id2entity = {v: k for k, v in entity2id.items()}
    num_entities = len(entity2id)
    
    # Create triples for all possible tails
    all_tails = torch.arange(num_entities, device=device)
    h_ids = torch.full((num_entities,), h_id, dtype=torch.long, device=device)
    r_ids = torch.full((num_entities,), r_id, dtype=torch.long, device=device)
    
    triples = torch.stack([h_ids, r_ids, all_tails], dim=1)
    
    # Score all triples
    scores = model(triples).cpu().numpy()
    
    # For TransE: lower is better, so we negate for ranking
    # For ComplEx/TriModel: higher is better
    if model_type == "TransE":
        # Sort ascending (lower distance = better)
        sorted_indices = np.argsort(scores)
    else:
        # Sort descending (higher score = better)
        sorted_indices = np.argsort(-scores)
    
    # Get top-k predictions
    results = []
    for idx in sorted_indices[:top_k]:
        entity = id2entity[idx]
        score = scores[idx]
        results.append((entity, float(score)))
    
    return results


@torch.no_grad()
def predict_heads(
    model,
    relation: str,
    tail: str,
    entity2id: dict,
    relation2id: dict,
    model_type: str,
    top_k: int = 10,
    device: str = "cpu"
) -> list:
    """
    Predict top-k head entities for a given (?, relation, tail).
    
    Args:
        model: Trained KGE model
        relation: Relation name
        tail: Tail entity name
        entity2id: Entity to ID mapping
        relation2id: Relation to ID mapping
        model_type: Type of model
        top_k: Number of top predictions to return
        device: Device to run on
        
    Returns:
        List of (entity, score) tuples sorted by score
    """
    if tail not in entity2id:
        raise ValueError(f"Unknown tail entity: {tail}")
    if relation not in relation2id:
        raise ValueError(f"Unknown relation: {relation}")
    
    r_id = relation2id[relation]
    t_id = entity2id[tail]
    
    id2entity = {v: k for k, v in entity2id.items()}
    num_entities = len(entity2id)
    
    # Create triples for all possible heads
    all_heads = torch.arange(num_entities, device=device)
    r_ids = torch.full((num_entities,), r_id, dtype=torch.long, device=device)
    t_ids = torch.full((num_entities,), t_id, dtype=torch.long, device=device)
    
    triples = torch.stack([all_heads, r_ids, t_ids], dim=1)
    
    # Score all triples
    scores = model(triples).cpu().numpy()
    
    # For TransE: lower is better
    # For ComplEx/TriModel: higher is better
    if model_type == "TransE":
        sorted_indices = np.argsort(scores)
    else:
        sorted_indices = np.argsort(-scores)
    
    # Get top-k predictions
    results = []
    for idx in sorted_indices[:top_k]:
        entity = id2entity[idx]
        score = scores[idx]
        results.append((entity, float(score)))
    
    return results


def print_results(results: list, model_type: str, file=None):
    """Pretty print scoring results."""
    def output(text=""):
        print(text)
        if file:
            file.write(text + "\n")
    
    output("\n" + "=" * 70)
    output("FACT SCORING RESULTS")
    output("=" * 70)
    
    if model_type == "TransE":
        output("(TransE: lower scores = more plausible)")
    else:
        output(f"({model_type}: higher scores = more plausible)")
    
    output("-" * 70)
    
    for fact, score, status in results:
        head, rel, tail = fact
        if score is not None:
            output(f"  ({head}, {rel}, {tail})")
            output(f"    Score: {score:.4f}")
        else:
            output(f"  ({head}, {rel}, {tail})")
            output(f"    Error: {status}")
        output()


def print_predictions(predictions: list, query_type: str, query: tuple, model_type: str, file=None):
    """Pretty print prediction results."""
    def output(text=""):
        print(text)
        if file:
            file.write(text + "\n")
    
    output("\n" + "=" * 70)
    
    if query_type == "tail":
        head, rel = query
        output(f"TOP TAIL PREDICTIONS for ({head}, {rel}, ?)")
    else:
        rel, tail = query
        output(f"TOP HEAD PREDICTIONS for (?, {rel}, {tail})")
    
    output("=" * 70)
    
    if model_type == "TransE":
        output("(TransE: lower scores = more plausible)")
    else:
        output(f"({model_type}: higher scores = more plausible)")
    
    output("-" * 70)
    
    for rank, (entity, score) in enumerate(predictions, 1):
        output(f"  {rank:3d}. {entity:40s} (score: {score:.4f})")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # Configuration
    DEVICE = "cpu"
    
    # Choose which model to use
    MODEL_PATH = "outputs_transe/transe_model.pt"  # TransE
    #MODEL_PATH = "outputs_complex/complex_model.pt"  # ComplEx
    #MODEL_PATH = "outputs_trimodel/trimodel_model.pt"  # TriModel
    
    # Output file path - automatically derived from MODEL_PATH
    OUTPUT_DIR = os.path.dirname(MODEL_PATH)
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "predictions.txt")
    
    # Load model
    print(f"Loading model from: {MODEL_PATH}")
    model, entity2id, relation2id, model_type = load_model(MODEL_PATH, DEVICE)
    
    print(f"  Entities:  {len(entity2id):,}")
    print(f"  Relations: {len(relation2id):,}")
    print(f"  Relations available: {list(relation2id.keys())}")
    
    # Open output file
    if OUTPUT_FILE:
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        outfile = open(OUTPUT_FILE, "w", encoding="utf-8")
        outfile.write(f"Fact Predictions - {model_type}\n")
        outfile.write(f"Model: {MODEL_PATH}\n")
        outfile.write(f"Entities: {len(entity2id):,}, Relations: {len(relation2id):,}\n")
        outfile.write("=" * 70 + "\n")
    else:
        outfile = None
    
    # ----------------------------
    # Example 1: Score specific facts
    # ----------------------------
    example_facts = [
        ["P61026", "INTERACT_WITH", "Q9H0K6"],
        ["DB00008", "DRUG_TARGET", "Q9H0K6"],
        ["DB00001", "DRUG_CATEGORY", "D000925"],
        # Add your own facts here
    ]
    
    print("\n" + "#" * 70)
    print("SCORING SPECIFIC FACTS")
    print("#" * 70)
    if outfile:
        outfile.write("\n" + "#" * 70 + "\n")
        outfile.write("SCORING SPECIFIC FACTS\n")
        outfile.write("#" * 70 + "\n")
    
    results = score_facts(
        model, 
        example_facts, 
        entity2id, 
        relation2id, 
        model_type,
        device=DEVICE
    )
    print_results(results, model_type, file=outfile)
    
    # ----------------------------
    # Example 2: Predict tail entities
    # ----------------------------
    print("\n" + "#" * 70)
    print("PREDICTING TAIL ENTITIES")
    print("#" * 70)
    if outfile:
        outfile.write("\n" + "#" * 70 + "\n")
        outfile.write("PREDICTING TAIL ENTITIES\n")
        outfile.write("#" * 70 + "\n")
    
    # Query: What drugs target protein Q9H0K6?
    # (?, DRUG_TARGET, Q9H0K6) - predict head
    try:
        head_entity = "DB00001"
        relation = "DRUG_CATEGORY"
        
        tail_predictions = predict_tails(
            model, head_entity, relation,
            entity2id, relation2id, model_type,
            top_k=10, device=DEVICE
        )
        print_predictions(tail_predictions, "tail", (head_entity, relation), model_type, file=outfile)
    except ValueError as e:
        print(f"Error: {e}")
    
    # ----------------------------
    # Example 3: Predict head entities
    # ----------------------------
    print("\n" + "#" * 70)
    print("PREDICTING HEAD ENTITIES")
    print("#" * 70)
    if outfile:
        outfile.write("\n" + "#" * 70 + "\n")
        outfile.write("PREDICTING HEAD ENTITIES\n")
        outfile.write("#" * 70 + "\n")
    
    try:
        relation = "DRUG_TARGET"
        tail_entity = "P00734"  # A protein
        
        head_predictions = predict_heads(
            model, relation, tail_entity,
            entity2id, relation2id, model_type,
            top_k=10, device=DEVICE
        )
        print_predictions(head_predictions, "head", (relation, tail_entity), model_type, file=outfile)
    except ValueError as e:
        print(f"Error: {e}")
    
    # Close output file
    if outfile:
        outfile.close()
        print(f"\nResults saved to: {OUTPUT_FILE}")
