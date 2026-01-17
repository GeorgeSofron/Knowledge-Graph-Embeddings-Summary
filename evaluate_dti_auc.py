"""
Drug-Target Interaction (DTI) Evaluation Script
================================================
Compute AUC-ROC and AUC-PR for DRUG_TARGET prediction across TransE, ComplEx, and TriModel.

This script:
1. Loads trained models
2. Extracts DRUG_TARGET triples from test set
3. Generates negative samples (random drug-protein pairs not in the graph)
4. Scores positive and negative samples
5. Computes AUC-ROC, AUC-PR, and related metrics
"""

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
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

from model import TransE, ComplEx, TriModel

# Output directory
OUTPUT_DIR = "outputs_dti_evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)


# ----------------------------
# Data Loading Functions
# ----------------------------
def load_triples(path):
    """Load triples from TSV file."""
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["source", "relation", "target"]
    )


def get_drug_target_triples(df):
    """Extract only DRUG_TARGET triples."""
    return df[df["relation"] == "DRUG_TARGET"].copy()


def get_all_entities_by_type(df):
    """Identify drugs and proteins based on naming conventions."""
    all_sources = set(df["source"].unique())
    all_targets = set(df["target"].unique())
    all_entities = all_sources | all_targets
    
    # Drugs start with "DB"
    drugs = {e for e in all_entities if str(e).startswith("DB")}
    
    # Proteins are targets of DRUG_TARGET relations (UniProt IDs like P12345, Q12345, O12345)
    dti_df = df[df["relation"] == "DRUG_TARGET"]
    proteins = set(dti_df["target"].unique())
    
    return drugs, proteins


def build_positive_set(train_df, valid_df, test_df):
    """Build set of all known DRUG_TARGET pairs."""
    positive_pairs = set()
    
    for df in [train_df, valid_df, test_df]:
        dti_df = get_drug_target_triples(df)
        for _, row in dti_df.iterrows():
            positive_pairs.add((row["source"], row["target"]))
    
    return positive_pairs


def generate_negative_samples(positive_pairs, drugs, proteins, n_negatives, seed=42):
    """
    Generate negative drug-target pairs by random sampling.
    Ensures negatives are not in the positive set.
    """
    np.random.seed(seed)
    
    drugs_list = list(drugs)
    proteins_list = list(proteins)
    
    negatives = set()
    attempts = 0
    max_attempts = n_negatives * 100  # Prevent infinite loop
    
    while len(negatives) < n_negatives and attempts < max_attempts:
        drug = np.random.choice(drugs_list)
        protein = np.random.choice(proteins_list)
        pair = (drug, protein)
        
        if pair not in positive_pairs and pair not in negatives:
            negatives.add(pair)
        
        attempts += 1
    
    return list(negatives)


# ----------------------------
# Model Loading Functions
# ----------------------------
def load_transe_model(checkpoint_path, device="cpu"):
    """Load trained TransE model."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = TransE(
        num_entities=ckpt["num_entities"],
        num_relations=ckpt["num_relations"],
        dim=ckpt["embedding_dim"],
        p_norm=ckpt["p_norm"],
    ).to(device)
    
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    return model, ckpt["entity2id"], ckpt["relation2id"]


def load_complex_model(checkpoint_path, device="cpu"):
    """Load trained ComplEx model."""
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


def load_trimodel_model(checkpoint_path, device="cpu"):
    """Load trained TriModel."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
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
# Scoring Functions
# ----------------------------
@torch.no_grad()
def score_transe(model, heads, relations, tails, device):
    """
    Score triples using TransE.
    TransE uses distance, so we negate it (lower distance = higher score).
    """
    h = model.ent(heads.to(device))
    r = model.rel(relations.to(device))
    t = model.ent(tails.to(device))
    
    # Distance score (lower is better for TransE)
    distance = torch.norm(h + r - t, p=model.p_norm, dim=1)
    
    # Convert to similarity score (higher is better)
    score = -distance
    
    return score.cpu().numpy()


@torch.no_grad()
def score_complex(model, heads, relations, tails, device):
    """
    Score triples using ComplEx.
    ComplEx uses similarity score directly (higher is better).
    """
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


@torch.no_grad()
def score_trimodel(model, heads, relations, tails, device):
    """
    Score triples using TriModel.
    TriModel uses similarity score directly (higher is better).
    """
    h_v1 = model.ent_v1(heads.to(device))
    h_v2 = model.ent_v2(heads.to(device))
    h_v3 = model.ent_v3(heads.to(device))
    
    r_v1 = model.rel_v1(relations.to(device))
    r_v2 = model.rel_v2(relations.to(device))
    r_v3 = model.rel_v3(relations.to(device))
    
    t_v1 = model.ent_v1(tails.to(device))
    t_v2 = model.ent_v2(tails.to(device))
    t_v3 = model.ent_v3(tails.to(device))
    
    # TriModel scoring function
    score = torch.sum(h_v1 * r_v1 * t_v1, dim=1) + \
            torch.sum(h_v2 * r_v2 * t_v2, dim=1) + \
            torch.sum(h_v3 * r_v3 * t_v3, dim=1)
    
    return score.cpu().numpy()


def score_pairs(model, model_type, pairs, relation_id, entity2id, device, batch_size=1024):
    """
    Score a list of (drug, protein) pairs.
    
    Args:
        model: The trained model
        model_type: 'transe', 'complex', or 'trimodel'
        pairs: List of (drug, protein) tuples
        relation_id: ID for DRUG_TARGET relation
        entity2id: Entity to ID mapping
        device: torch device
        batch_size: Batch size for scoring
        
    Returns:
        numpy array of scores
    """
    # Filter pairs to only include entities known to the model
    valid_pairs = []
    for drug, protein in pairs:
        if drug in entity2id and protein in entity2id:
            valid_pairs.append((entity2id[drug], relation_id, entity2id[protein]))
    
    if len(valid_pairs) == 0:
        return np.array([])
    
    # Convert to tensors
    heads = torch.tensor([p[0] for p in valid_pairs], dtype=torch.long)
    relations = torch.tensor([p[1] for p in valid_pairs], dtype=torch.long)
    tails = torch.tensor([p[2] for p in valid_pairs], dtype=torch.long)
    
    # Score in batches
    all_scores = []
    
    for i in range(0, len(valid_pairs), batch_size):
        batch_h = heads[i:i+batch_size]
        batch_r = relations[i:i+batch_size]
        batch_t = tails[i:i+batch_size]
        
        if model_type == 'transe':
            scores = score_transe(model, batch_h, batch_r, batch_t, device)
        elif model_type == 'complex':
            scores = score_complex(model, batch_h, batch_r, batch_t, device)
        elif model_type == 'trimodel':
            scores = score_trimodel(model, batch_h, batch_r, batch_t, device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        all_scores.extend(scores)
    
    return np.array(all_scores)


# ----------------------------
# Evaluation Functions
# ----------------------------
def compute_metrics(y_true, y_scores):
    """Compute AUC-ROC, AUC-PR, and other metrics."""
    metrics = {}
    
    # Safety check
    if len(y_true) == 0 or len(y_scores) == 0:
        return {'AUC-ROC': 0, 'AUC-PR': 0, 'Best_F1': 0, 
                'Precision@Best_F1': 0, 'Recall@Best_F1': 0, 'Best_Threshold': 0}
    
    # AUC-ROC
    metrics['AUC-ROC'] = roc_auc_score(y_true, y_scores)
    
    # AUC-PR (Average Precision)
    metrics['AUC-PR'] = average_precision_score(y_true, y_scores)
    
    # Find optimal threshold using F1
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    # Note: thresholds has length len(precision) - 1
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-10)
    if len(f1_scores) > 0:
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
    else:
        best_threshold = 0.5
    
    # Metrics at optimal threshold
    y_pred = (y_scores >= best_threshold).astype(int)
    metrics['Best_F1'] = f1_score(y_true, y_pred)
    metrics['Precision@Best_F1'] = precision_score(y_true, y_pred)
    metrics['Recall@Best_F1'] = recall_score(y_true, y_pred)
    metrics['Best_Threshold'] = best_threshold
    
    return metrics


def evaluate_model(model, model_type, model_name, test_dti_pairs, all_positive_pairs, 
                   drugs, proteins, entity2id, relation2id, device, neg_ratio=10):
    """
    Evaluate a single model on DTI prediction task.
    
    Args:
        model: Trained model
        model_type: 'transe', 'complex', or 'trimodel'
        model_name: Display name for the model
        test_dti_pairs: List of (drug, protein) test positive pairs
        all_positive_pairs: Set of all known positive pairs (for filtering negatives)
        drugs: Set of all drug entities
        proteins: Set of all protein entities
        entity2id: Entity to ID mapping
        relation2id: Relation to ID mapping
        device: torch device
        neg_ratio: Ratio of negatives to positives
        
    Returns:
        Dictionary of metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")
    
    # Filter test pairs to entities known by this model
    valid_test_pairs = [(d, p) for d, p in test_dti_pairs if d in entity2id and p in entity2id]
    print(f"Valid test DTI pairs: {len(valid_test_pairs)}")
    
    if len(valid_test_pairs) == 0:
        print("No valid test pairs found!")
        return None
    
    # Generate negative samples
    n_negatives = len(valid_test_pairs) * neg_ratio
    print(f"Generating {n_negatives} negative samples...")
    
    # Filter drugs and proteins to those known by the model
    model_drugs = {d for d in drugs if d in entity2id}
    model_proteins = {p for p in proteins if p in entity2id}
    
    negative_pairs = generate_negative_samples(
        all_positive_pairs, model_drugs, model_proteins, n_negatives
    )
    print(f"Generated {len(negative_pairs)} negative samples")
    
    # Get relation ID for DRUG_TARGET
    if "DRUG_TARGET" not in relation2id:
        print("DRUG_TARGET relation not found in model!")
        return None
    
    drug_target_rel_id = relation2id["DRUG_TARGET"]
    
    # Score positive pairs
    print("Scoring positive pairs...")
    positive_scores = score_pairs(
        model, model_type, valid_test_pairs, drug_target_rel_id, entity2id, device
    )
    
    # Score negative pairs
    print("Scoring negative pairs...")
    negative_scores = score_pairs(
        model, model_type, negative_pairs, drug_target_rel_id, entity2id, device
    )
    
    # Safety check for empty arrays
    if len(positive_scores) == 0:
        print("Warning: No valid positive scores!")
        return None
    if len(negative_scores) == 0:
        print("Warning: No valid negative scores!")
        return None
    
    # Combine scores and labels
    y_scores = np.concatenate([positive_scores, negative_scores])
    y_true = np.concatenate([
        np.ones(len(positive_scores)),
        np.zeros(len(negative_scores))
    ])
    
    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(y_true, y_scores)
    
    # Store additional info
    metrics['n_positives'] = len(positive_scores)
    metrics['n_negatives'] = len(negative_scores)
    
    return metrics, y_true, y_scores


def plot_roc_curves(results, save_path):
    """Plot ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = {'TransE': '#3498db', 'ComplEx': '#e74c3c', 'TriModel': '#2ecc71'}
    
    for model_name, data in results.items():
        if data is None:
            continue
        
        y_true, y_scores = data['y_true'], data['y_scores']
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = data['metrics']['AUC-ROC']
        
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})', 
                color=colors.get(model_name, 'gray'), linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves for Drug-Target Prediction', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_pr_curves(results, save_path):
    """Plot Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = {'TransE': '#3498db', 'ComplEx': '#e74c3c', 'TriModel': '#2ecc71'}
    
    for model_name, data in results.items():
        if data is None:
            continue
        
        y_true, y_scores = data['y_true'], data['y_scores']
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = data['metrics']['AUC-PR']
        
        ax.plot(recall, precision, label=f'{model_name} (AP = {ap:.4f})', 
                color=colors.get(model_name, 'gray'), linewidth=2)
    
    # Baseline (random classifier)
    baseline = sum(results[list(results.keys())[0]]['y_true']) / len(results[list(results.keys())[0]]['y_true'])
    ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1, label=f'Random (AP = {baseline:.4f})')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves for Drug-Target Prediction', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_metrics_comparison(results, save_path):
    """Plot bar chart comparing AUC-ROC and AUC-PR."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    model_names = [name for name, data in results.items() if data is not None]
    auc_roc = [results[name]['metrics']['AUC-ROC'] for name in model_names]
    auc_pr = [results[name]['metrics']['AUC-PR'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, auc_roc, width, label='AUC-ROC', color='#3498db')
    bars2 = ax.bar(x + width/2, auc_pr, width, label='AUC-PR', color='#e74c3c')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Drug-Target Interaction Prediction: AUC Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ----------------------------
# Main
# ----------------------------
def main():
    print("=" * 70)
    print("DRUG-TARGET INTERACTION PREDICTION EVALUATION")
    print("AUC-ROC and AUC-PR Metrics")
    print("=" * 70)
    
    DEVICE = "cpu"
    NEG_RATIO = 80  # 80 negatives per positive
    
    # Model paths
    MODEL_PATHS = {
        'TransE': ('outputs_transe/transe_model.pt', 'transe', 'data/transe'),
        'ComplEx': ('outputs_complex/complex_model.pt', 'complex', 'data/complex'),
        'TriModel': ('outputs_trimodel/trimodel_model.pt', 'trimodel', 'data/trimodel'),
    }
    
    # Load all data to build complete positive set
    print("\n📂 Loading data...")
    
    # Try to find the appropriate data directory
    data_dirs = ['data/trimodel', 'data/transe', 'data/complex', 'data']
    
    train_df = None
    valid_df = None
    test_df = None
    
    for data_dir in data_dirs:
        try:
            train_df = load_triples(f"{data_dir}/train.txt")
            test_df = load_triples(f"{data_dir}/test.txt")
            try:
                valid_df = load_triples(f"{data_dir}/valid.txt")
            except:
                valid_df = pd.DataFrame(columns=["source", "relation", "target"])
            print(f"Loaded data from {data_dir}")
            break
        except:
            continue
    
    if train_df is None:
        print("Error: Could not load data files!")
        return
    
    # Get all positive pairs
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
    
    # Evaluate each model
    results = {}
    
    for model_name, (model_path, model_type, data_dir) in MODEL_PATHS.items():
        if not os.path.exists(model_path):
            print(f"\n⚠️ {model_name} model not found at {model_path}")
            results[model_name] = None
            continue
        
        try:
            # Load model
            print(f"\n📦 Loading {model_name}...")
            if model_type == 'transe':
                model, entity2id, relation2id = load_transe_model(model_path, DEVICE)
            elif model_type == 'complex':
                model, entity2id, relation2id = load_complex_model(model_path, DEVICE)
            elif model_type == 'trimodel':
                model, entity2id, relation2id = load_trimodel_model(model_path, DEVICE)
            
            # Load model-specific test data if available
            try:
                model_test_df = load_triples(f"{data_dir}/test.txt")
                model_test_dti = get_drug_target_triples(model_test_df)
                model_test_pairs = list(zip(model_test_dti["source"], model_test_dti["target"]))
                print(f"Model-specific test DTI pairs: {len(model_test_pairs)}")
            except:
                model_test_pairs = test_dti_pairs
            
            # Evaluate
            result = evaluate_model(
                model, model_type, model_name,
                model_test_pairs, all_positive_pairs,
                drugs, proteins, entity2id, relation2id, DEVICE, NEG_RATIO
            )
            
            if result is not None:
                metrics, y_true, y_scores = result
                results[model_name] = {
                    'metrics': metrics,
                    'y_true': y_true,
                    'y_scores': y_scores
                }
            else:
                results[model_name] = None
                
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = None
    
    # Print results summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    metrics_df = []
    for model_name, data in results.items():
        if data is None:
            continue
        
        m = data['metrics']
        print(f"\n{model_name}:")
        print(f"  AUC-ROC:          {m['AUC-ROC']:.4f}")
        print(f"  AUC-PR:           {m['AUC-PR']:.4f}")
        print(f"  Best F1:          {m['Best_F1']:.4f}")
        print(f"  Precision@Best:   {m['Precision@Best_F1']:.4f}")
        print(f"  Recall@Best:      {m['Recall@Best_F1']:.4f}")
        print(f"  Positives/Negs:   {m['n_positives']}/{m['n_negatives']}")
        
        metrics_df.append({
            'Model': model_name,
            'AUC-ROC': m['AUC-ROC'],
            'AUC-PR': m['AUC-PR'],
            'Best_F1': m['Best_F1'],
            'Precision': m['Precision@Best_F1'],
            'Recall': m['Recall@Best_F1'],
            'n_positives': m['n_positives'],
            'n_negatives': m['n_negatives']
        })
    
    # Save results
    if metrics_df:
        df = pd.DataFrame(metrics_df)
        df.to_csv(os.path.join(OUTPUT_DIR, "dti_metrics.csv"), index=False)
        print(f"\nSaved: {os.path.join(OUTPUT_DIR, 'dti_metrics.csv')}")
        
        # Generate plots
        print("\n🎨 Generating plots...")
        plot_roc_curves(results, os.path.join(OUTPUT_DIR, "figures", "roc_curves.png"))
        plot_pr_curves(results, os.path.join(OUTPUT_DIR, "figures", "pr_curves.png"))
        plot_metrics_comparison(results, os.path.join(OUTPUT_DIR, "figures", "auc_comparison.png"))
        
        # Save detailed report
        with open(os.path.join(OUTPUT_DIR, "dti_evaluation_report.txt"), "w") as f:
            f.write("DRUG-TARGET INTERACTION PREDICTION EVALUATION\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Negative Ratio: {NEG_RATIO}:1\n\n")
            
            for model_name, data in results.items():
                if data is None:
                    f.write(f"{model_name}: Model not available\n\n")
                    continue
                
                m = data['metrics']
                f.write(f"{model_name}:\n")
                f.write(f"  AUC-ROC:          {m['AUC-ROC']:.4f}\n")
                f.write(f"  AUC-PR:           {m['AUC-PR']:.4f}\n")
                f.write(f"  Best F1:          {m['Best_F1']:.4f}\n")
                f.write(f"  Precision@Best:   {m['Precision@Best_F1']:.4f}\n")
                f.write(f"  Recall@Best:      {m['Recall@Best_F1']:.4f}\n")
                f.write(f"  Best Threshold:   {m['Best_Threshold']:.4f}\n")
                f.write(f"  Test Positives:   {m['n_positives']}\n")
                f.write(f"  Test Negatives:   {m['n_negatives']}\n")
                f.write("\n")
        
        print(f"Saved: {os.path.join(OUTPUT_DIR, 'dti_evaluation_report.txt')}")
    
    print("\n" + "=" * 70)
    print("✅ DTI Evaluation Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
