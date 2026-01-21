"""
ComplEx Evaluation Script
=========================
Evaluate trained ComplEx model on link prediction task.
Uses filtered ranking protocol with MRR, Hits@1, Hits@3, Hits@10.
Also computes ROC and PR curves for binary classification.
"""

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score, average_precision_score

from model import ComplEx  # Shared model definition


# ----------------------------
# Load trained model
# ----------------------------
def load_model(checkpoint_path, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = ComplEx(
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


def encode_triples(df, entity2id, relation2id):
    """Encode triples DataFrame to tensor of IDs."""
    h = df["source"].map(entity2id).to_numpy()
    r = df["relation"].map(relation2id).to_numpy()
    t = df["target"].map(entity2id).to_numpy()
    triples = np.stack([h, r, t], axis=1)
    return torch.tensor(triples, dtype=torch.long)


# ----------------------------
# Ranking functions (filtered) - Optimized with vectorization
# ----------------------------
@torch.no_grad()
def compute_ranks_batch(model, test_triples, all_true, num_entities, device, batch_size=256):
    """
    Compute filtered ranks for a batch of test triples.
    
    For ComplEx, higher scores are better (unlike TransE where lower is better).
    
    Args:
        model: Trained ComplEx model
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
    
    for start in tqdm(range(0, n_test, batch_size), desc="Evaluating batches"):
        batch = test_triples[start:start + batch_size].to(device)
        batch_h = batch[:, 0]
        batch_r = batch[:, 1]
        batch_t = batch[:, 2]
        
        # --- Tail prediction: score all entities as potential tails ---
        tail_scores = model.score_all_tails(batch_h, batch_r)  # (batch, num_entities)
        tail_scores = tail_scores.cpu().numpy()
        
        # --- Head prediction: score all entities as potential heads ---
        head_scores = model.score_all_heads(batch_r, batch_t)  # (batch, num_entities)
        head_scores = head_scores.cpu().numpy()
        
        # Apply filtering and compute ranks
        for i in range(batch.size(0)):
            h, r, t = batch[i].cpu().tolist()
            
            # Filter tail scores (higher is better for ComplEx)
            tail_sc = tail_scores[i].copy()
            for e in range(num_entities):
                if (h, r, e) in all_true and e != t:
                    tail_sc[e] = -np.inf  # Filter out other true triples
            # Rank: how many entities have HIGHER score than the correct tail
            tail_rank = int((tail_sc > tail_sc[t]).sum() + 1)
            all_tail_ranks.append(tail_rank)
            
            # Filter head scores
            head_sc = head_scores[i].copy()
            for e in range(num_entities):
                if (e, r, t) in all_true and e != h:
                    head_sc[e] = -np.inf
            head_rank = int((head_sc > head_sc[h]).sum() + 1)
            all_head_ranks.append(head_rank)
    
    return np.array(all_head_ranks), np.array(all_tail_ranks)


# ----------------------------
# Evaluation
# ----------------------------
def evaluate(model, test_triples, all_true, num_entities, device, batch_size=256):
    """
    Evaluate model on test triples using batched computation.
    
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


def save_evaluation_results(metrics: dict, output_path: str):
    """Save evaluation results to file."""
    with open(output_path, "w") as f:
        f.write("ComplEx Evaluation Results\n")
        f.write("=" * 40 + "\n\n")
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\n")
    print(f"Results saved to: {output_path}")


# ----------------------------
# ROC and PR Curve Functions
# ----------------------------
@torch.no_grad()
def generate_binary_classification_data(model, positive_triples, entity2id, relation2id, 
                                         all_true, device, neg_ratio=10):
    """
    Generate positive and negative samples for ROC/PR curve computation.
    
    Args:
        model: Trained ComplEx model
        positive_triples: DataFrame of positive test triples
        entity2id: Entity to ID mapping
        relation2id: Relation to ID mapping
        all_true: Set of all true triples (for avoiding false negatives)
        device: torch device
        neg_ratio: Number of negative samples per positive
        
    Returns:
        labels: Array of binary labels (1=positive, 0=negative)
        scores: Array of model scores
    """
    model.eval()
    labels = []
    scores = []
    
    all_entities = list(entity2id.values())
    
    for _, row in tqdm(positive_triples.iterrows(), total=len(positive_triples), desc="Generating samples"):
        h = entity2id[row["source"]]
        r = relation2id[row["relation"]]
        t = entity2id[row["target"]]
        
        # Positive sample - create triple tensor (batch_size=1, 3)
        pos_triple = torch.tensor([[h, r, t]], dtype=torch.long, device=device)
        pos_score = model(pos_triple).item()
        labels.append(1)
        scores.append(pos_score)
        
        # Negative samples (corrupt tail)
        neg_count = 0
        attempts = 0
        while neg_count < neg_ratio and attempts < neg_ratio * 10:
            neg_t = np.random.choice(all_entities)
            if (h, r, neg_t) not in all_true:
                neg_triple = torch.tensor([[h, r, neg_t]], dtype=torch.long, device=device)
                neg_score = model(neg_triple).item()
                labels.append(0)
                scores.append(neg_score)
                neg_count += 1
            attempts += 1
    
    return np.array(labels), np.array(scores)


def compute_roc_pr_metrics(labels, scores):
    """
    Compute ROC and PR curves and their AUC values.
    
    Returns:
        Dictionary with curve data and metrics
    """
    # ROC curve
    fpr, tpr, roc_thresholds = roc_curve(labels, scores)
    roc_auc = roc_auc_score(labels, scores)
    
    # PR curve
    precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
    pr_auc = average_precision_score(labels, scores)
    
    # Best F1 score
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]
    best_precision = precision[best_f1_idx]
    best_recall = recall[best_f1_idx]
    
    return {
        "roc": {"fpr": fpr, "tpr": tpr, "thresholds": roc_thresholds},
        "pr": {"precision": precision, "recall": recall, "thresholds": pr_thresholds},
        "auc_roc": roc_auc,
        "auc_pr": pr_auc,
        "best_f1": best_f1,
        "best_precision": best_precision,
        "best_recall": best_recall
    }


def plot_roc_curve(roc_data, auc_value, output_path):
    """Plot and save ROC curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(roc_data["fpr"], roc_data["tpr"], color='#3498db', lw=2, 
             label=f'ComplEx (AUC = {auc_value:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - ComplEx Model', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to: {output_path}")


def plot_pr_curve(pr_data, auc_value, output_path):
    """Plot and save Precision-Recall curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(pr_data["recall"], pr_data["precision"], color='#e74c3c', lw=2,
             label=f'ComplEx (AUC = {auc_value:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve - ComplEx Model', fontsize=14, fontweight='bold')
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"PR curve saved to: {output_path}")


def plot_combined_curves(roc_data, pr_data, auc_roc, auc_pr, output_path):
    """Plot both ROC and PR curves side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC curve
    ax = axes[0]
    ax.plot(roc_data["fpr"], roc_data["tpr"], color='#3498db', lw=2,
            label=f'ComplEx (AUC = {auc_roc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    
    # PR curve
    ax = axes[1]
    ax.plot(pr_data["recall"], pr_data["precision"], color='#e74c3c', lw=2,
            label=f'ComplEx (AUC = {auc_pr:.4f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Combined curves saved to: {output_path}")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    DEVICE = "cpu"  # change to "cuda" if available
    BATCH_SIZE = 256
    NEG_RATIO = 10  # Negative samples per positive for ROC/PR

    MODEL_PATH = "outputs_complex/complex_model.pt"
    OUTPUT_DIR = "outputs_complex"

    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    model, entity2id, relation2id = load_model(MODEL_PATH, DEVICE)
    num_entities = len(entity2id)
    print(f"Loaded ComplEx model: {num_entities} entities, {len(relation2id)} relations")

    # Load data
    train_df = load_triples("data/complex/train.txt")
    test_df = load_triples("data/complex/test.txt")
    
    # Filter test data to only include known entities/relations
    test_df = test_df[
        (test_df["source"].isin(entity2id.keys())) &
        (test_df["relation"].isin(relation2id.keys())) &
        (test_df["target"].isin(entity2id.keys()))
    ].reset_index(drop=True)
    
    print(f"Filtered test set: {len(test_df)} triples")
    
    # Encode test triples
    test_triples = encode_triples(test_df, entity2id, relation2id)

    # Build set of all known triples for filtered ranking
    all_true = triples_to_id_set(train_df, entity2id, relation2id)
    all_true |= triples_to_id_set(test_df, entity2id, relation2id)
    
    # Try to add validation triples if available
    valid_path = "data/complex/valid.txt"
    if os.path.exists(valid_path):
        valid_df = load_triples(valid_path)
        valid_df = valid_df[
            (valid_df["source"].isin(entity2id.keys())) &
            (valid_df["relation"].isin(relation2id.keys())) &
            (valid_df["target"].isin(entity2id.keys()))
        ]
        all_true |= triples_to_id_set(valid_df, entity2id, relation2id)
        print(f"Added {len(valid_df)} validation triples to filter set")

    print(f"Total true triples for filtering: {len(all_true)}")

    # Evaluate link prediction metrics
    print("\nEvaluating ComplEx model (Link Prediction)...")
    metrics = evaluate(model, test_triples, all_true, num_entities, DEVICE, BATCH_SIZE)

    print("\n" + "=" * 40)
    print("ComplEx Link Prediction Results")
    print("=" * 40)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print("=" * 40)
    
    # Save link prediction results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_evaluation_results(metrics, os.path.join(OUTPUT_DIR, "Evaluation.txt"))
    
    # Generate ROC and PR curves
    print(f"\nGenerating ROC/PR curves (NEG_RATIO={NEG_RATIO})...")
    labels, scores = generate_binary_classification_data(
        model, test_df, entity2id, relation2id, all_true, DEVICE, neg_ratio=NEG_RATIO
    )
    
    # Compute metrics
    curve_metrics = compute_roc_pr_metrics(labels, scores)
    
    print("\n" + "=" * 40)
    print("ComplEx Binary Classification Results")
    print("=" * 40)
    print(f"AUC-ROC: {curve_metrics['auc_roc']:.4f}")
    print(f"AUC-PR:  {curve_metrics['auc_pr']:.4f}")
    print(f"Best F1: {curve_metrics['best_f1']:.4f}")
    print(f"  Precision @ Best F1: {curve_metrics['best_precision']:.4f}")
    print(f"  Recall @ Best F1:    {curve_metrics['best_recall']:.4f}")
    print("=" * 40)
    
    # Create figures directory
    figures_dir = os.path.join(OUTPUT_DIR, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Plot curves
    plot_roc_curve(curve_metrics["roc"], curve_metrics["auc_roc"], 
                   os.path.join(figures_dir, f"roc_curve_neg{NEG_RATIO}.png"))
    plot_pr_curve(curve_metrics["pr"], curve_metrics["auc_pr"],
                  os.path.join(figures_dir, f"pr_curve_neg{NEG_RATIO}.png"))
    plot_combined_curves(curve_metrics["roc"], curve_metrics["pr"],
                         curve_metrics["auc_roc"], curve_metrics["auc_pr"],
                         os.path.join(figures_dir, f"roc_pr_curves_neg{NEG_RATIO}.png"))
    
    # Save binary classification results
    with open(os.path.join(OUTPUT_DIR, f"binary_classification_neg{NEG_RATIO}.txt"), "w") as f:
        f.write("ComplEx Binary Classification Results\n")
        f.write(f"NEG_RATIO: {NEG_RATIO}\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"AUC-ROC: {curve_metrics['auc_roc']:.4f}\n")
        f.write(f"AUC-PR:  {curve_metrics['auc_pr']:.4f}\n")
        f.write(f"Best F1: {curve_metrics['best_f1']:.4f}\n")
        f.write(f"Precision @ Best F1: {curve_metrics['best_precision']:.4f}\n")
        f.write(f"Recall @ Best F1:    {curve_metrics['best_recall']:.4f}\n")
    
    print(f"\nAll results saved to {OUTPUT_DIR}/")
    print("Done!")
