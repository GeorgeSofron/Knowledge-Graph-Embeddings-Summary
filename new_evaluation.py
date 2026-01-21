"""
DTI Evaluation (Binary Classification) - Fixed & Fair Version
============================================================

Fixes vs original:
1) Entity typing (drugs/proteins) built from train+valid+test (not train only)
2) Per-drug negative sampling (negatives paired to each positive's drug)
3) Optional degree-matched negatives (by target degree in TRAIN)
4) Same negative set reused across models (for fair comparison)
5) Robust PR baseline (doesn't crash if some models missing)
6) Warnings when requested negatives can't be reached
7) Reports how many test positives were dropped due to missing entities (transductive KGE)

Assumes:
- Triples TSV files: train.txt / valid.txt / test.txt with columns: head, relation, tail
- DTI relation name: "DRUG_TARGET"
- Entities: drugs start with "DB" (DrugBank IDs)
- Proteins: targets of DRUG_TARGET edges (typically UniProt IDs)

Models:
- TransE, ComplEx, TriModel imported from model.py
"""

import os
import math
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional

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


# ----------------------------
# Config
# ----------------------------

@dataclass
class EvalConfig:
    dti_relation: str = "DRUG_TARGET"
    device: str = "cpu"
    neg_ratio: int = 10                 # negatives per positive
    neg_sampling: str = "degree_matched"  # "random" or "degree_matched"
    seed: int = 42
    batch_size: int = 4096

    # Output
    output_dir: str = "outputs_dti_evaluation_fixed"


# ----------------------------
# IO helpers
# ----------------------------

def load_triples(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=None, names=["source", "relation", "target"])


def get_dti_pairs(df: pd.DataFrame, rel: str) -> List[Tuple[str, str]]:
    dti = df[df["relation"] == rel]
    return list(zip(dti["source"].astype(str), dti["target"].astype(str)))


def build_positive_set(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame, rel: str) -> Set[Tuple[str, str]]:
    pos = set()
    for df in [train_df, valid_df, test_df]:
        for pair in get_dti_pairs(df, rel):
            pos.add(pair)
    return pos


def get_all_entities_by_type(all_df: pd.DataFrame, rel: str) -> Tuple[Set[str], Set[str]]:
    """
    Robust typing:
    - Drugs: any entity starting with 'DB'
    - Proteins: any entity ever appearing as tail in DTI edges across ALL splits
    """
    all_sources = set(all_df["source"].astype(str).unique())
    all_targets = set(all_df["target"].astype(str).unique())
    all_entities = all_sources | all_targets

    drugs = {e for e in all_entities if e.startswith("DB")}

    dti_df = all_df[all_df["relation"] == rel]
    proteins = set(dti_df["target"].astype(str).unique())

    return drugs, proteins


# ----------------------------
# Model loaders (same as yours)
# ----------------------------

def load_transe_model(checkpoint_path, device="cpu"):
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
# Scoring (same semantics, batched)
# ----------------------------

@torch.no_grad()
def score_transe(model, heads, relations, tails, device):
    h = model.ent(heads.to(device))
    r = model.rel(relations.to(device))
    t = model.ent(tails.to(device))
    distance = torch.norm(h + r - t, p=model.p_norm, dim=1)
    return (-distance).cpu().numpy()


@torch.no_grad()
def score_complex(model, heads, relations, tails, device):
    h_re = model.ent_re(heads.to(device))
    h_im = model.ent_im(heads.to(device))
    r_re = model.rel_re(relations.to(device))
    r_im = model.rel_im(relations.to(device))
    t_re = model.ent_re(tails.to(device))
    t_im = model.ent_im(tails.to(device))
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
    h_v1 = model.ent_v1(heads.to(device))
    h_v2 = model.ent_v2(heads.to(device))
    h_v3 = model.ent_v3(heads.to(device))

    r_v1 = model.rel_v1(relations.to(device))
    r_v2 = model.rel_v2(relations.to(device))
    r_v3 = model.rel_v3(relations.to(device))

    t_v1 = model.ent_v1(tails.to(device))
    t_v2 = model.ent_v2(tails.to(device))
    t_v3 = model.ent_v3(tails.to(device))

    score = (h_v1 * r_v1 * t_v1).sum(dim=1) + (h_v2 * r_v2 * t_v2).sum(dim=1) + (h_v3 * r_v3 * t_v3).sum(dim=1)
    return score.cpu().numpy()


def score_pairs(model, model_type, pairs, relation_id, entity2id, device, batch_size=4096) -> np.ndarray:
    triples = []
    for d, p in pairs:
        if d in entity2id and p in entity2id:
            triples.append((entity2id[d], relation_id, entity2id[p]))

    if not triples:
        return np.array([])

    h = torch.tensor([x[0] for x in triples], dtype=torch.long)
    r = torch.tensor([x[1] for x in triples], dtype=torch.long)
    t = torch.tensor([x[2] for x in triples], dtype=torch.long)

    out = []
    for i in range(0, len(triples), batch_size):
        bh, br, bt = h[i:i+batch_size], r[i:i+batch_size], t[i:i+batch_size]
        if model_type == "transe":
            out.extend(score_transe(model, bh, br, bt, device))
        elif model_type == "complex":
            out.extend(score_complex(model, bh, br, bt, device))
        elif model_type == "trimodel":
            out.extend(score_trimodel(model, bh, br, bt, device))
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    return np.array(out)


# ----------------------------
# Negative sampling (fixed)
# ----------------------------

def compute_target_degree(train_pairs: List[Tuple[str, str]]) -> Dict[str, int]:
    deg = {}
    for _, p in train_pairs:
        deg[p] = deg.get(p, 0) + 1
    return deg


def _build_degree_bins(proteins: List[str], p_deg: Dict[str, int], bin_size: int = 5) -> Dict[int, List[str]]:
    bins = {}
    for p in proteins:
        b = p_deg.get(p, 0) // bin_size
        bins.setdefault(b, []).append(p)
    return bins


def generate_negatives_per_drug(
    pos_pairs: List[Tuple[str, str]],
    all_pos_set: Set[Tuple[str, str]],
    proteins_universe: List[str],
    neg_ratio: int,
    rng: np.random.Generator,
    mode: str = "random",
    p_deg: Optional[Dict[str, int]] = None,
    bin_size: int = 5,
    max_attempts_factor: int = 200
) -> List[Tuple[str, str]]:
    """
    For each positive (d, p), sample neg_ratio proteins for the SAME drug d.
    - mode="random": uniform proteins_universe
    - mode="degree_matched": sample proteins with similar train-degree to p

    Returns list length approx len(pos_pairs) * neg_ratio (can be smaller if graph dense).
    """
    proteins_arr = np.array(proteins_universe, dtype=object)

    degree_bins = None
    if mode == "degree_matched":
        if p_deg is None:
            raise ValueError("p_deg required for degree_matched")
        degree_bins = _build_degree_bins(proteins_universe, p_deg, bin_size=bin_size)

    negatives: List[Tuple[str, str]] = []
    needed = len(pos_pairs) * neg_ratio
    max_attempts = needed * max_attempts_factor
    attempts = 0

    for (d, p) in pos_pairs:
        local_need = neg_ratio
        while local_need > 0 and attempts < max_attempts:
            attempts += 1

            if mode == "random":
                p_neg = str(rng.choice(proteins_arr))
            else:
                # degree matched by p's bin; widen if needed
                pb = p_deg.get(p, 0) // bin_size
                cand = degree_bins.get(pb, [])
                if len(cand) < 20:
                    cand = []
                    for bb in [pb, pb-1, pb+1, pb-2, pb+2]:
                        cand.extend(degree_bins.get(bb, []))
                    if len(cand) < 20:
                        cand = proteins_universe
                p_neg = str(rng.choice(np.array(cand, dtype=object)))

            pair = (d, p_neg)
            if pair not in all_pos_set:
                negatives.append(pair)
                local_need -= 1

            if attempts >= max_attempts:
                break

        if attempts >= max_attempts:
            break

    if len(negatives) < needed:
        print(f"⚠️ Warning: requested {needed} negatives, generated {len(negatives)}. "
              f"(Graph may be dense or universe too small)")

    return negatives


# ----------------------------
# Metrics + plots (robust)
# ----------------------------

def compute_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0 or len(y_scores) == 0:
        return {'AUC-ROC': 0, 'AUC-PR': 0, 'Best_F1': 0,
                'Precision@Best_F1': 0, 'Recall@Best_F1': 0, 'Best_Threshold': 0}

    metrics = {}
    metrics['AUC-ROC'] = float(roc_auc_score(y_true, y_scores))
    metrics['AUC-PR'] = float(average_precision_score(y_true, y_scores))

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-10)

    if len(f1_scores) > 0:
        best_idx = int(np.argmax(f1_scores))
        best_threshold = float(thresholds[best_idx])
    else:
        best_threshold = 0.5

    y_pred = (y_scores >= best_threshold).astype(int)
    metrics['Best_F1'] = float(f1_score(y_true, y_pred))
    metrics['Precision@Best_F1'] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics['Recall@Best_F1'] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics['Best_Threshold'] = float(best_threshold)
    return metrics


def _first_non_none(results: Dict) -> Optional[Dict]:
    for _, v in results.items():
        if v is not None:
            return v
    return None


def plot_roc_curves(results: Dict, save_path: str):
    fig, ax = plt.subplots(figsize=(8, 8))
    for model_name, data in results.items():
        if data is None:
            continue
        y_true, y_scores = data['y_true'], data['y_scores']
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = data['metrics']['AUC-ROC']
        ax.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.4f})', linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves (DTI)')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_pr_curves(results: Dict, save_path: str):
    fig, ax = plt.subplots(figsize=(8, 8))
    for model_name, data in results.items():
        if data is None:
            continue
        y_true, y_scores = data['y_true'], data['y_scores']
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = data['metrics']['AUC-PR']
        ax.plot(recall, precision, label=f'{model_name} (AP={ap:.4f})', linewidth=2)

    first = _first_non_none(results)
    if first is not None:
        y_true0 = first['y_true']
        baseline = float(y_true0.mean())
        ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1, label=f'Random baseline={baseline:.4f}')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves (DTI)')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_metrics_comparison(results: Dict, save_path: str):
    model_names = [k for k, v in results.items() if v is not None]
    if not model_names:
        return

    auc_roc = [results[n]['metrics']['AUC-ROC'] for n in model_names]
    auc_pr = [results[n]['metrics']['AUC-PR'] for n in model_names]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(model_names))
    width = 0.35
    ax.bar(x - width/2, auc_roc, width, label='AUC-ROC')
    ax.bar(x + width/2, auc_pr, width, label='AUC-PR')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('DTI Prediction: AUC Comparison')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ----------------------------
# Core evaluation
# ----------------------------

def evaluate_one_model(
    model, model_type: str, model_name: str,
    test_pos_pairs: List[Tuple[str, str]],
    neg_pairs_fixed: List[Tuple[str, str]],
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
    cfg: EvalConfig
):
    if cfg.dti_relation not in relation2id:
        print(f"⚠️ {model_name}: relation {cfg.dti_relation} not in relation2id")
        return None

    rel_id = relation2id[cfg.dti_relation]

    # Transductive filtering:
    pos_valid = [(d, p) for (d, p) in test_pos_pairs if d in entity2id and p in entity2id]
    dropped = len(test_pos_pairs) - len(pos_valid)
    print(f"{model_name}: test positives={len(test_pos_pairs)} valid={len(pos_valid)} dropped={dropped}")

    if not pos_valid:
        return None

    # Negatives: also filter to entity vocab (so each model scores same *intended* negatives)
    neg_valid = [(d, p) for (d, p) in neg_pairs_fixed if d in entity2id and p in entity2id]
    if not neg_valid:
        print(f"⚠️ {model_name}: no valid negatives after filtering to entity vocab")
        return None

    pos_scores = score_pairs(model, model_type, pos_valid, rel_id, entity2id, cfg.device, batch_size=cfg.batch_size)
    neg_scores = score_pairs(model, model_type, neg_valid, rel_id, entity2id, cfg.device, batch_size=cfg.batch_size)

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return None

    y_scores = np.concatenate([pos_scores, neg_scores])
    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])

    metrics = compute_metrics(y_true, y_scores)
    metrics["n_positives"] = int(len(pos_scores))
    metrics["n_negatives"] = int(len(neg_scores))
    metrics["dropped_test_positives"] = int(dropped)

    return {"metrics": metrics, "y_true": y_true, "y_scores": y_scores}


# ----------------------------
# Main
# ----------------------------

def main():
    cfg = EvalConfig()
    np_rng = np.random.default_rng(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "figures"), exist_ok=True)

    print("=" * 70)
    print("DTI EVALUATION (FIXED)")
    print("=" * 70)
    print(f"neg_ratio={cfg.neg_ratio}  neg_sampling={cfg.neg_sampling}  seed={cfg.seed}")
    print()

    # Locate dataset directory
    data_dirs = ['data/trimodel', 'data/transe', 'data/complex', 'data']
    train_df = valid_df = test_df = None
    data_dir_used = None

    for d in data_dirs:
        try:
            train_df = load_triples(f"{d}/train.txt")
            test_df = load_triples(f"{d}/test.txt")
            try:
                valid_df = load_triples(f"{d}/valid.txt")
            except Exception:
                valid_df = pd.DataFrame(columns=["source", "relation", "target"])
            data_dir_used = d
            print(f"Loaded data from {d}")
            break
        except Exception:
            continue

    if train_df is None:
        print("❌ Could not load train/test data from expected folders.")
        return

    # Build global sets from ALL splits (fix)
    all_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    drugs, proteins = get_all_entities_by_type(all_df, cfg.dti_relation)
    print(f"Drugs={len(drugs)} Proteins={len(proteins)} (typed from all splits)")

    # Positives
    all_positive_pairs = build_positive_set(train_df, valid_df, test_df, cfg.dti_relation)
    test_pos_pairs = get_dti_pairs(test_df, cfg.dti_relation)
    train_pos_pairs = get_dti_pairs(train_df, cfg.dti_relation)

    print(f"All known positives (all splits): {len(all_positive_pairs)}")
    print(f"Test positives: {len(test_pos_pairs)}")

    # Target degree (from TRAIN only)
    p_deg = compute_target_degree(train_pos_pairs)

    # Canonical protein universe (all proteins), canonical positives = test_pos_pairs
    proteins_universe = sorted(list(proteins))

    # IMPORTANT: sample negatives ONCE (fixed) so models are comparable
    neg_pairs_fixed = generate_negatives_per_drug(
        pos_pairs=test_pos_pairs,
        all_pos_set=all_positive_pairs,
        proteins_universe=proteins_universe,
        neg_ratio=cfg.neg_ratio,
        rng=np_rng,
        mode=cfg.neg_sampling,
        p_deg=p_deg if cfg.neg_sampling == "degree_matched" else None
    )
    print(f"Fixed negatives generated: {len(neg_pairs_fixed)}")

    # Models to evaluate
    model_paths = {
        'TransE': ('outputs_transe/transe_model.pt', 'transe', 'data/transe'),
        'ComplEx': ('outputs_complex/complex_model.pt', 'complex', 'data/complex'),
        'TriModel': ('outputs_trimodel/trimodel_model.pt', 'trimodel', 'data/trimodel'),
    }

    results = {}

    for model_name, (ckpt_path, model_type, model_data_dir) in model_paths.items():
        if not os.path.exists(ckpt_path):
            print(f"⚠️ {model_name}: checkpoint not found at {ckpt_path}")
            results[model_name] = None
            continue

        print("\n" + "-" * 70)
        print(f"Loading {model_name}...")

        if model_type == "transe":
            model, entity2id, relation2id = load_transe_model(ckpt_path, cfg.device)
        elif model_type == "complex":
            model, entity2id, relation2id = load_complex_model(ckpt_path, cfg.device)
        elif model_type == "trimodel":
            model, entity2id, relation2id = load_trimodel_model(ckpt_path, cfg.device)
        else:
            raise ValueError(model_type)

        # Optionally use model-specific test split if present (keeps your original behavior)
        try:
            mtest_df = load_triples(f"{model_data_dir}/test.txt")
            mtest_pos = get_dti_pairs(mtest_df, cfg.dti_relation)
            print(f"{model_name}: model-specific test positives={len(mtest_pos)}")
        except Exception:
            mtest_pos = test_pos_pairs

        # NOTE: If model-specific test positives differ, fairness is impacted.
        # For strict fairness, comment the try/except above and always use test_pos_pairs.
        out = evaluate_one_model(
            model, model_type, model_name,
            test_pos_pairs=mtest_pos,
            neg_pairs_fixed=neg_pairs_fixed,
            entity2id=entity2id,
            relation2id=relation2id,
            cfg=cfg
        )
        results[model_name] = out

    # Summary + save
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    rows = []
    for model_name, data in results.items():
        if data is None:
            print(f"{model_name}: (no result)")
            continue
        m = data["metrics"]
        print(f"\n{model_name}:")
        print(f"  AUC-ROC: {m['AUC-ROC']:.4f}")
        print(f"  AUC-PR : {m['AUC-PR']:.4f}")
        print(f"  Best F1: {m['Best_F1']:.4f}  (thr={m['Best_Threshold']:.4f})")
        print(f"  Precision/Recall@Best: {m['Precision@Best_F1']:.4f}/{m['Recall@Best_F1']:.4f}")
        print(f"  Pos/Neg: {m['n_positives']}/{m['n_negatives']}  dropped_pos={m['dropped_test_positives']}")

        rows.append({
            "Model": model_name,
            "AUC-ROC": m["AUC-ROC"],
            "AUC-PR": m["AUC-PR"],
            "Best_F1": m["Best_F1"],
            "Best_Threshold": m["Best_Threshold"],
            "Precision@Best_F1": m["Precision@Best_F1"],
            "Recall@Best_F1": m["Recall@Best_F1"],
            "n_positives": m["n_positives"],
            "n_negatives": m["n_negatives"],
            "dropped_test_positives": m["dropped_test_positives"],
            "neg_ratio": cfg.neg_ratio,
            "neg_sampling": cfg.neg_sampling,
            "seed": cfg.seed,
            "data_dir_used": data_dir_used,
        })

    if rows:
        out_csv = os.path.join(cfg.output_dir, "dti_metrics.csv")
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"\nSaved: {out_csv}")

        plot_roc_curves(results, os.path.join(cfg.output_dir, "figures", "roc_curves.png"))
        plot_pr_curves(results, os.path.join(cfg.output_dir, "figures", "pr_curves.png"))
        plot_metrics_comparison(results, os.path.join(cfg.output_dir, "figures", "auc_comparison.png"))

        report_path = os.path.join(cfg.output_dir, "dti_evaluation_report.txt")
        with open(report_path, "w") as f:
            f.write("DTI EVALUATION (FIXED)\n")
            f.write("=" * 60 + "\n")
            f.write(f"DTI relation: {cfg.dti_relation}\n")
            f.write(f"Negative ratio: {cfg.neg_ratio}:1\n")
            f.write(f"Negative sampling: {cfg.neg_sampling}\n")
            f.write(f"Seed: {cfg.seed}\n\n")
            for r in rows:
                f.write(f"{r['Model']}:\n")
                f.write(f"  AUC-ROC: {r['AUC-ROC']:.4f}\n")
                f.write(f"  AUC-PR : {r['AUC-PR']:.4f}\n")
                f.write(f"  Best F1: {r['Best_F1']:.4f} (thr={r['Best_Threshold']:.4f})\n")
                f.write(f"  Precision/Recall@Best: {r['Precision@Best_F1']:.4f}/{r['Recall@Best_F1']:.4f}\n")
                f.write(f"  Pos/Neg: {r['n_positives']}/{r['n_negatives']}\n")
                f.write(f"  Dropped test positives (OOV): {r['dropped_test_positives']}\n\n")

        print(f"Saved: {report_path}")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
