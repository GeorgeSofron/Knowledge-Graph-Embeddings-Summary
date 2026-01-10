import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from model import TransE  # Shared model definition


# ----------------------------
# Load trained model
# ----------------------------
def load_model(checkpoint_path, device="cpu"):
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
    
    Args:
        model: Trained TransE model
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
    
    # Pre-compute all entity embeddings
    all_ent_emb = model.ent.weight.data  # (num_entities, dim)
    
    for start in tqdm(range(0, n_test, batch_size), desc="Evaluating batches"):
        batch = test_triples[start:start + batch_size].to(device)
        batch_h = batch[:, 0]
        batch_r = batch[:, 1]
        batch_t = batch[:, 2]
        
        h_emb = model.ent(batch_h)  # (batch, dim)
        r_emb = model.rel(batch_r)  # (batch, dim)
        t_emb = model.ent(batch_t)  # (batch, dim)
        
        # --- Tail prediction: score all entities as potential tails ---
        # (h + r) for each triple in batch
        hr = h_emb + r_emb  # (batch, dim)
        
        # Compute distance to ALL entities: ||h + r - t'||
        # hr: (batch, dim), all_ent_emb: (num_entities, dim)
        # Result: (batch, num_entities)
        tail_scores = torch.cdist(hr.unsqueeze(1), all_ent_emb.unsqueeze(0), p=model.p_norm).squeeze(1)
        tail_scores = tail_scores.cpu().numpy()
        
        # --- Head prediction: score all entities as potential heads ---
        # (t - r) for each triple in batch
        t_minus_r = t_emb - r_emb  # (batch, dim)
        
        # Compute distance to ALL entities: ||h' - (t - r)|| = ||h' + r - t||
        head_scores = torch.cdist(t_minus_r.unsqueeze(1), all_ent_emb.unsqueeze(0), p=model.p_norm).squeeze(1)
        head_scores = head_scores.cpu().numpy()
        
        # Apply filtering and compute ranks
        for i in range(batch.size(0)):
            h, r, t = batch[i].cpu().tolist()
            
            # Filter tail scores
            tail_sc = tail_scores[i].copy()
            for e in range(num_entities):
                if (h, r, e) in all_true and e != t:
                    tail_sc[e] = np.inf
            tail_rank = int((tail_sc < tail_sc[t]).sum() + 1)
            all_tail_ranks.append(tail_rank)
            
            # Filter head scores
            head_sc = head_scores[i].copy()
            for e in range(num_entities):
                if (e, r, t) in all_true and e != h:
                    head_sc[e] = np.inf
            head_rank = int((head_sc < head_sc[h]).sum() + 1)
            all_head_ranks.append(head_rank)
    
    return np.array(all_head_ranks), np.array(all_tail_ranks)


# ----------------------------
# Evaluation loop (optimized)
# ----------------------------
def evaluate(model, test_triples, all_true, num_entities, device, batch_size=256):
    """
    Evaluate model on test triples using batched computation.
    
    Args:
        model: Trained TransE model
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
    }


# ----------------------------
# Main
# ----------------------------
def encode_triples(df, entity2id, relation2id):
    """Encode triples DataFrame to tensor of IDs."""
    h = df["source"].map(entity2id).to_numpy()
    r = df["relation"].map(relation2id).to_numpy()
    t = df["target"].map(entity2id).to_numpy()
    triples = np.stack([h, r, t], axis=1)
    return torch.tensor(triples, dtype=torch.long)


if __name__ == "__main__":
    DEVICE = "cpu"  # change to "cuda" if available
    BATCH_SIZE = 256  # Batch size for evaluation

    MODEL_PATH = "outputs_transe/transe_model.pt"
    DATA_DIR = "data/transe"
    OUTPUT_PATH = "outputs_transe/Evaluation.txt"

    # Load model
    model, entity2id, relation2id = load_model(MODEL_PATH, DEVICE)
    num_entities = len(entity2id)

    # Load data
    train_df = load_triples(f"{DATA_DIR}/train.txt")
    test_df = load_triples(f"{DATA_DIR}/test.txt")
    
    # Filter test data to only include entities/relations known to the model
    test_df = test_df[
        (test_df["source"].isin(entity2id.keys())) &
        (test_df["relation"].isin(relation2id.keys())) &
        (test_df["target"].isin(entity2id.keys()))
    ].reset_index(drop=True)
    
    print(f"Filtered test set: {len(test_df)} triples (from original dataset)")
    
    # Encode test triples to tensor
    test_triples = encode_triples(test_df, entity2id, relation2id)

    # True triples for filtering (include BOTH train and test for proper filtered ranking)
    # This ensures other true test triples don't unfairly inflate ranks
    all_true = triples_to_id_set(train_df, entity2id, relation2id)
    all_true |= triples_to_id_set(test_df, entity2id, relation2id)
    
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

    # Evaluate with batched computation
    metrics = evaluate(
        model,
        test_triples,
        all_true,
        num_entities,
        DEVICE,
        batch_size=BATCH_SIZE
    )

    print("\nEvaluation results (filtered):")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    # Save results
    import os
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write("TransE Evaluation Results (Filtered Ranking)\n")
        f.write("=" * 50 + "\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    print(f"\nResults saved to {OUTPUT_PATH}")
