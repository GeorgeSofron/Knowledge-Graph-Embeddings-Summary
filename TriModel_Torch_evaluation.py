"""
TriModel Evaluation Script
==========================
Evaluate a trained TriModel on knowledge graph link prediction task.
Computes filtered metrics: MRR, Hits@1, Hits@3, Hits@10
"""

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from model import TriModel  # Shared model definition


# ----------------------------
# Load trained model
# ----------------------------
def load_model(checkpoint_path, device="cpu"):
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


# ----------------------------
# Load triples
# ----------------------------
def load_triples(path):
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["source", "relation", "target"]
    )


def triples_to_id_set(df, entity2id, relation2id):
    return set(
        zip(
            df["source"].map(entity2id),
            df["relation"].map(relation2id),
            df["target"].map(entity2id),
        )
    )


# ----------------------------
# Ranking functions (filtered) - Optimized with vectorization
# ----------------------------
@torch.no_grad()
def compute_ranks_batch(model, test_triples, all_true, num_entities, device, batch_size=256):
    """
    Compute filtered ranks for a batch of test triples.
    Significantly faster than row-by-row evaluation.
    
    For TriModel, higher scores indicate more plausible triples,
    so we rank by descending score (unlike TransE which uses distance).
    
    Args:
        model: Trained TriModel
        test_triples: Tensor of shape (N, 3) with [head, relation, tail] IDs
        all_true: Set of all true triples for filtering
        num_entities: Total number of entities
        device: torch device
        batch_size: Number of triples to evaluate at once
        
    Returns:
        Tuple of (head_ranks, tail_ranks) arrays
    """
    model.eval()
    n_test = test_triples.size(0)
    
    all_tail_ranks = []
    all_head_ranks = []
    
    # Pre-compute all entity embeddings (3 vectors each)
    all_ent_v1 = model.ent_v1.weight.data  # (num_entities, dim)
    all_ent_v2 = model.ent_v2.weight.data
    all_ent_v3 = model.ent_v3.weight.data
    
    for start in tqdm(range(0, n_test, batch_size), desc="Evaluating batches"):
        batch = test_triples[start:start + batch_size].to(device)
        batch_h = batch[:, 0]
        batch_r = batch[:, 1]
        batch_t = batch[:, 2]
        
        # Get embeddings for batch
        h_v1 = model.ent_v1(batch_h)  # (batch, dim)
        h_v2 = model.ent_v2(batch_h)
        h_v3 = model.ent_v3(batch_h)
        
        r_v1 = model.rel_v1(batch_r)
        r_v2 = model.rel_v2(batch_r)
        r_v3 = model.rel_v3(batch_r)
        
        t_v1 = model.ent_v1(batch_t)
        t_v2 = model.ent_v2(batch_t)
        t_v3 = model.ent_v3(batch_t)
        
        # --- Tail prediction: score all entities as potential tails ---
        # Score: h1*r1*t3 + h2*r2*t2 + h3*r3*t1
        # For all candidate tails t', we compute:
        # (h_v1 * r_v1) @ all_ent_v3.T + (h_v2 * r_v2) @ all_ent_v2.T + (h_v3 * r_v3) @ all_ent_v1.T
        hr1 = h_v1 * r_v1  # (batch, dim)
        hr2 = h_v2 * r_v2
        hr3 = h_v3 * r_v3
        
        tail_scores = (hr1 @ all_ent_v3.T +  # (batch, num_entities)
                       hr2 @ all_ent_v2.T +
                       hr3 @ all_ent_v1.T)
        tail_scores = tail_scores.cpu().numpy()
        
        # --- Head prediction: score all entities as potential heads ---
        # For all candidate heads h', we compute:
        # all_ent_v1 @ (r1*t3).T + all_ent_v2 @ (r2*t2).T + all_ent_v3 @ (r3*t1).T
        r1t3 = r_v1 * t_v3  # (batch, dim)
        r2t2 = r_v2 * t_v2
        r3t1 = r_v3 * t_v1
        
        head_scores = (all_ent_v1 @ r1t3.T +  # (num_entities, batch)
                       all_ent_v2 @ r2t2.T +
                       all_ent_v3 @ r3t1.T)
        head_scores = head_scores.T.cpu().numpy()  # (batch, num_entities)
        
        # Apply filtering and compute ranks
        for i in range(batch.size(0)):
            h, r, t = batch[i].cpu().tolist()
            
            # Filter tail scores (set other true tails to -inf so they don't count)
            tail_sc = tail_scores[i].copy()
            for e in range(num_entities):
                if (h, r, e) in all_true and e != t:
                    tail_sc[e] = -np.inf
            # Higher score = better, so count how many have HIGHER score
            tail_rank = int((tail_sc > tail_sc[t]).sum() + 1)
            all_tail_ranks.append(tail_rank)
            
            # Filter head scores
            head_sc = head_scores[i].copy()
            for e in range(num_entities):
                if (e, r, t) in all_true and e != h:
                    head_sc[e] = -np.inf
            # Higher score = better, so count how many have HIGHER score
            head_rank = int((head_sc > head_sc[h]).sum() + 1)
            all_head_ranks.append(head_rank)
    
    return np.array(all_head_ranks), np.array(all_tail_ranks)


# ----------------------------
# Evaluation loop (optimized)
# ----------------------------
def evaluate(model, test_triples, all_true, num_entities, device, batch_size=256):
    """
    Evaluate model on test triples using batched computation.
    
    Args:
        model: Trained TriModel
        test_triples: Tensor of (N, 3) with encoded test triples
        all_true: Set of all known true triples
        num_entities: Total number of entities
        device: torch device
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary with MRR, Hits@1, Hits@3, Hits@10
    """
    head_ranks, tail_ranks = compute_ranks_batch(
        model, test_triples, all_true, num_entities, device, batch_size
    )
    
    # Combine head and tail ranks
    all_ranks = np.concatenate([head_ranks, tail_ranks])
    
    return {
        "MRR": float(np.mean(1.0 / all_ranks)),
        "Hits@1": float((all_ranks <= 1).mean()),
        "Hits@3": float((all_ranks <= 3).mean()),
        "Hits@10": float((all_ranks <= 10).mean()),
        "Mean Rank": float(np.mean(all_ranks)),
    }


# ----------------------------
# Encode triples
# ----------------------------
def encode_triples(df, entity2id, relation2id):
    """Encode triples DataFrame to tensor of IDs."""
    h = df["source"].map(entity2id).to_numpy()
    r = df["relation"].map(relation2id).to_numpy()
    t = df["target"].map(entity2id).to_numpy()
    triples = np.stack([h, r, t], axis=1)
    return torch.tensor(triples, dtype=torch.long)


# ----------------------------
# Save evaluation results
# ----------------------------
def save_results(metrics, output_path="outputs_trimodel/Evaluation.txt"):
    """Save evaluation metrics to file."""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write("TriModel Evaluation Results (Filtered Ranking)\n")
        f.write("=" * 50 + "\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    
    print(f"\nResults saved to {output_path}")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    DEVICE = "cpu"  # change to "cuda" if available
    BATCH_SIZE = 256  # Batch size for evaluation

    MODEL_PATH = "outputs_trimodel/trimodel_model.pt"
    DATA_DIR = "data/trimodel"
    OUTPUT_PATH = "outputs_trimodel/Evaluation.txt"

    # Load model
    print("Loading model...")
    model, entity2id, relation2id = load_model(MODEL_PATH, DEVICE)
    num_entities = len(entity2id)
    num_relations = len(relation2id)
    
    print(f"Model loaded: {num_entities} entities, {num_relations} relations")
    print(f"Embedding dim: {model.dim} (total: {model.dim * 3} per entity/relation)")

    # Load data
    train_df = load_triples(f"{DATA_DIR}/train.txt")
    test_df = load_triples(f"{DATA_DIR}/test.txt")
    
    print(f"\nTrain triples: {len(train_df)}")
    print(f"Test triples: {len(test_df)}")
    
    # Filter test data to only include entities/relations known to the model
    test_df_filtered = test_df[
        (test_df["source"].isin(entity2id.keys())) &
        (test_df["relation"].isin(relation2id.keys())) &
        (test_df["target"].isin(entity2id.keys()))
    ].reset_index(drop=True)
    
    print(f"Filtered test set: {len(test_df_filtered)} triples")
    
    # Encode test triples to tensor
    test_triples = encode_triples(test_df_filtered, entity2id, relation2id)

    # True triples for filtering (include BOTH train and test for proper filtered ranking)
    all_true = triples_to_id_set(train_df, entity2id, relation2id)
    all_true |= triples_to_id_set(test_df_filtered, entity2id, relation2id)
    
    # Also load validation set if available
    try:
        valid_df = load_triples(f"{DATA_DIR}/valid.txt")
        valid_df_filtered = valid_df[
            (valid_df["source"].isin(entity2id.keys())) &
            (valid_df["relation"].isin(relation2id.keys())) &
            (valid_df["target"].isin(entity2id.keys()))
        ]
        all_true |= triples_to_id_set(valid_df_filtered, entity2id, relation2id)
        print(f"Validation triples loaded: {len(valid_df_filtered)}")
    except FileNotFoundError:
        print("No validation file found, proceeding without it.")

    print(f"\nTotal true triples for filtering: {len(all_true)}")

    # Evaluate with batched computation
    print("\nStarting evaluation...")
    metrics = evaluate(
        model,
        test_triples,
        all_true,
        num_entities,
        DEVICE,
        batch_size=BATCH_SIZE
    )

    print("\n" + "=" * 50)
    print("Evaluation Results (Filtered Ranking)")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    # Save results
    save_results(metrics, OUTPUT_PATH)
