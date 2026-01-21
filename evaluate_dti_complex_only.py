"""
Drug-Target Interaction (DTI) Evaluation Script for ComplEx Only
================================================================
Compute AUC-ROC and AUC-PR for DRUG_TARGET prediction using only the ComplEx model.

This script:
1. Loads the trained ComplEx model
2. Extracts DRUG_TARGET triples from test set
3. Generates negative samples (random drug-protein pairs not in the graph)
4. Scores positive and negative samples
5. Computes AUC-ROC, AUC-PR, and related metrics
"""

import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score
)
import matplotlib.pyplot as plt

from model import ComplEx

OUTPUT_DIR = "outputs_dti_evaluation_complex"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)

# ----------------------------
# Data Loading Functions
# ----------------------------
def load_triples(path):
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["source", "relation", "target"]
    )

def get_drug_target_triples(df):
    return df[df["relation"] == "DRUG_TARGET"].copy()

def get_all_entities_by_type(df):
    all_sources = set(df["source"].unique())
    all_targets = set(df["target"].unique())
    all_entities = all_sources | all_targets
    drugs = {e for e in all_entities if str(e).startswith("DB")}
    dti_df = df[df["relation"] == "DRUG_TARGET"]
    proteins = set(dti_df["target"].unique())
    return drugs, proteins

def build_positive_set(train_df, valid_df, test_df):
    positive_pairs = set()
    for df in [train_df, valid_df, test_df]:
        dti_df = get_drug_target_triples(df)
        for _, row in dti_df.iterrows():
            positive_pairs.add((row["source"], row["target"]))
    return positive_pairs

def generate_negative_samples(positive_pairs, drugs, proteins, n_negatives, seed=42):
    np.random.seed(seed)
    drugs_list = list(drugs)
    proteins_list = list(proteins)
    negatives = set()
    attempts = 0
    max_attempts = n_negatives * 100
    while len(negatives) < n_negatives and attempts < max_attempts:
        drug = np.random.choice(drugs_list)
        protein = np.random.choice(proteins_list)
        pair = (drug, protein)
        if pair not in positive_pairs and pair not in negatives:
            negatives.add(pair)
        attempts += 1
    return list(negatives)

# ----------------------------
# Model Loading Function
# ----------------------------
def load_complex_model(checkpoint_path, device="cpu"):
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
# Scoring Function
# ----------------------------
@torch.no_grad()
def score_complex(model, heads, relations, tails, device):
    h_re = model.ent_re(heads.to(device))
    h_im = model.ent_im(heads.to(device))
    r_re = model.rel_re(relations.to(device))
    r_im = model.rel_im(relations.to(device))
    t_re = model.ent_re(tails.to(device))
    t_im = model.ent_im(tails.to(device))
    # ComplEx scoring function
    score = torch.sum(
        r_re * h_re * t_re +
        r_re * h_im * t_im +
        r_im * h_re * t_im -
        r_im * h_im * t_re,
        dim=1
    )
    return score.cpu().numpy()

def score_pairs(model, pairs, relation_id, entity2id, device, batch_size=1024):
    valid_pairs = []
    for drug, protein in pairs:
        if drug in entity2id and protein in entity2id:
            valid_pairs.append((entity2id[drug], relation_id, entity2id[protein]))
    if len(valid_pairs) == 0:
        return np.array([])
    heads = torch.tensor([p[0] for p in valid_pairs], dtype=torch.long)
    relations = torch.tensor([p[1] for p in valid_pairs], dtype=torch.long)
    tails = torch.tensor([p[2] for p in valid_pairs], dtype=torch.long)
    all_scores = []
    for i in range(0, len(valid_pairs), batch_size):
        batch_h = heads[i:i+batch_size]
        batch_r = relations[i:i+batch_size]
        batch_t = tails[i:i+batch_size]
        scores = score_complex(model, batch_h, batch_r, batch_t, device)
        all_scores.extend(scores)
    return np.array(all_scores)

# ----------------------------
# Evaluation Functions
# ----------------------------
def compute_metrics(y_true, y_scores):
    metrics = {}
    if len(y_true) == 0 or len(y_scores) == 0:
        return {'AUC-ROC': 0, 'AUC-PR': 0, 'Best_F1': 0,
                'Precision@Best_F1': 0, 'Recall@Best_F1': 0, 'Best_Threshold': 0}
    metrics['AUC-ROC'] = roc_auc_score(y_true, y_scores)
    metrics['AUC-PR'] = average_precision_score(y_true, y_scores)
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-10)
    if len(f1_scores) > 0:
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
    else:
        best_threshold = 0.5
    y_pred = (y_scores >= best_threshold).astype(int)
    metrics['Best_F1'] = f1_score(y_true, y_pred)
    metrics['Precision@Best_F1'] = precision_score(y_true, y_pred)
    metrics['Recall@Best_F1'] = recall_score(y_true, y_pred)
    metrics['Best_Threshold'] = best_threshold
    return metrics

def plot_roc_curve(y_true, y_scores, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, label=f'ComplEx (AUC = {auc:.4f})', color='#e74c3c', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve for Drug-Target Prediction (ComplEx)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_pr_curve(y_true, y_scores, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    baseline = sum(y_true) / len(y_true)
    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, label=f'ComplEx (AP = {ap:.4f})', color='#e74c3c', linewidth=2)
    plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1, label=f'Random (AP = {baseline:.4f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve for Drug-Target Prediction (ComplEx)', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

# ----------------------------
# Main
# ----------------------------
def main():
    print("=" * 70)
    print("DRUG-TARGET INTERACTION PREDICTION EVALUATION (ComplEx Only)")
    print("AUC-ROC and AUC-PR Metrics")
    print("=" * 70)
    DEVICE = "cpu"
    NEG_RATIO = 80
    MODEL_PATH = 'outputs_complex/complex_model.pt'
    DATA_DIR = 'data/complex'
    # Load data
    print(f"\n📂 Loading data from {DATA_DIR} ...")
    train_df = load_triples(f"{DATA_DIR}/train.txt")
    test_df = load_triples(f"{DATA_DIR}/test.txt")
    try:
        valid_df = load_triples(f"{DATA_DIR}/valid.txt")
    except:
        valid_df = pd.DataFrame(columns=["source", "relation", "target"])
    # Build positive set
    all_positive_pairs = build_positive_set(train_df, valid_df, test_df)
    print(f"Total known DRUG_TARGET pairs: {len(all_positive_pairs)}")
    # Get test DTI pairs
    test_dti_df = get_drug_target_triples(test_df)
    test_dti_pairs = list(zip(test_dti_df["source"], test_dti_df["target"]))
    print(f"Test DRUG_TARGET pairs: {len(test_dti_pairs)}")
    # Get all drugs and proteins
    drugs, proteins = get_all_entities_by_type(train_df)
    print(f"Unique drugs: {len(drugs)}")
    print(f"Unique proteins: {len(proteins)}")
    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"\n⚠️ ComplEx model not found at {MODEL_PATH}")
        return
    print(f"\n📦 Loading ComplEx model ...")
    model, entity2id, relation2id = load_complex_model(MODEL_PATH, DEVICE)
    # Evaluate
    if "DRUG_TARGET" not in relation2id:
        print("DRUG_TARGET relation not found in model!")
        return
    drug_target_rel_id = relation2id["DRUG_TARGET"]
    # Filter test pairs to entities known by this model
    valid_test_pairs = [(d, p) for d, p in test_dti_pairs if d in entity2id and p in entity2id]
    print(f"Valid test DTI pairs: {len(valid_test_pairs)}")
    if len(valid_test_pairs) == 0:
        print("No valid test pairs found!")
        return
    # Generate negative samples
    n_negatives = len(valid_test_pairs) * NEG_RATIO
    print(f"Generating {n_negatives} negative samples...")
    model_drugs = {d for d in drugs if d in entity2id}
    model_proteins = {p for p in proteins if p in entity2id}
    negative_pairs = generate_negative_samples(
        all_positive_pairs, model_drugs, model_proteins, n_negatives
    )
    print(f"Generated {len(negative_pairs)} negative samples")
    # Score positive pairs
    print("Scoring positive pairs...")
    positive_scores = score_pairs(
        model, valid_test_pairs, drug_target_rel_id, entity2id, DEVICE
    )
    # Score negative pairs
    print("Scoring negative pairs...")
    negative_scores = score_pairs(
        model, negative_pairs, drug_target_rel_id, entity2id, DEVICE
    )
    if len(positive_scores) == 0:
        print("Warning: No valid positive scores!")
        return
    if len(negative_scores) == 0:
        print("Warning: No valid negative scores!")
        return
    y_scores = np.concatenate([positive_scores, negative_scores])
    y_true = np.concatenate([
        np.ones(len(positive_scores)),
        np.zeros(len(negative_scores))
    ])
    print("Computing metrics...")
    metrics = compute_metrics(y_true, y_scores)
    metrics['n_positives'] = len(positive_scores)
    metrics['n_negatives'] = len(negative_scores)
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (ComplEx)")
    print("=" * 70)
    print(f"  AUC-ROC:          {metrics['AUC-ROC']:.4f}")
    print(f"  AUC-PR:           {metrics['AUC-PR']:.4f}")
    print(f"  Best F1:          {metrics['Best_F1']:.4f}")
    print(f"  Precision@Best:   {metrics['Precision@Best_F1']:.4f}")
    print(f"  Recall@Best:      {metrics['Recall@Best_F1']:.4f}")
    print(f"  Positives/Negs:   {metrics['n_positives']}/{metrics['n_negatives']}")
    # Save results
    df = pd.DataFrame([metrics])
    df.to_csv(os.path.join(OUTPUT_DIR, f"dti_metrics_complex_neg{NEG_RATIO}.csv"), index=False)
    print(f"\nSaved: {os.path.join(OUTPUT_DIR, f'dti_metrics_complex_neg{NEG_RATIO}.csv')}")
    # Generate plots
    print("\n🎨 Generating plots...")
    plot_roc_curve(y_true, y_scores, os.path.join(OUTPUT_DIR, "figures", f"roc_curve_complex_neg{NEG_RATIO}.png"))
    plot_pr_curve(y_true, y_scores, os.path.join(OUTPUT_DIR, "figures", f"pr_curve_complex_neg{NEG_RATIO}.png"))
    # Save detailed report
    with open(os.path.join(OUTPUT_DIR, f"dti_evaluation_report_complex_neg{NEG_RATIO}.txt"), "w") as f:
        f.write("DRUG-TARGET INTERACTION PREDICTION EVALUATION (ComplEx)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Negative Ratio: {NEG_RATIO}:1\n\n")
        f.write(f"AUC-ROC:          {metrics['AUC-ROC']:.4f}\n")
        f.write(f"AUC-PR:           {metrics['AUC-PR']:.4f}\n")
        f.write(f"Best F1:          {metrics['Best_F1']:.4f}\n")
        f.write(f"Precision@Best:   {metrics['Precision@Best_F1']:.4f}\n")
        f.write(f"Recall@Best:      {metrics['Recall@Best_F1']:.4f}\n")
        f.write(f"Best Threshold:   {metrics['Best_Threshold']:.4f}\n")
        f.write(f"Test Positives:   {metrics['n_positives']}\n")
        f.write(f"Test Negatives:   {metrics['n_negatives']}\n")
    print(f"Saved: {os.path.join(OUTPUT_DIR, f'dti_evaluation_report_complex_neg{NEG_RATIO}.txt')}")
    print("\n" + "=" * 70)
    print("✅ DTI Evaluation Complete! (ComplEx)")
    print("=" * 70)

if __name__ == "__main__":
    main()
