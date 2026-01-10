"""
Model Comparison Script
=======================
Compare TransE, ComplEx, and TriModel performance on knowledge graph link prediction.
Generates comparison tables and visualizations.
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Output directory for comparison results
OUTPUT_DIR = "outputs_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)


def parse_evaluation_file(filepath):
    """Parse evaluation results from a text file."""
    metrics = {}
    
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return metrics
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Parse metrics using regex
    patterns = {
        'MRR': r'MRR:\s*([\d.]+)',
        'Hits@1': r'Hits@1:\s*([\d.]+)',
        'Hits@3': r'Hits@3:\s*([\d.]+)',
        'Hits@10': r'Hits@10:\s*([\d.]+)',
        'Mean Rank': r'Mean Rank:\s*([\d.]+)',
    }
    
    for metric, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            metrics[metric] = float(match.group(1))
    
    return metrics


def load_all_results():
    """Load evaluation results from all three models."""
    models = {
        'TransE': 'outputs_transe/Evaluation.txt',
        'ComplEx': 'outputs_complex/Evaluation.txt',
        'TriModel': 'outputs_trimodel/Evaluation.txt',
    }
    
    results = {}
    for model_name, filepath in models.items():
        results[model_name] = parse_evaluation_file(filepath)
    
    return results


def load_training_losses():
    """Load training loss curves from all models."""
    models = {
        'TransE': 'outputs_transe/training_loss.csv',
        'ComplEx': 'outputs_complex/training_loss.csv',
        'TriModel': 'outputs_trimodel/training_loss.csv',
    }
    
    losses = {}
    for model_name, filepath in models.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            losses[model_name] = df
    
    return losses


def get_model_sizes():
    """Get model file sizes as a proxy for model complexity."""
    models = {
        'TransE': 'outputs_transe/transe_model.pt',
        'ComplEx': 'outputs_complex/complex_model.pt',
        'TriModel': 'outputs_trimodel/trimodel_model.pt',
    }
    
    sizes = {}
    for model_name, filepath in models.items():
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            sizes[model_name] = size_mb
    
    return sizes


def create_comparison_table(results):
    """Create a comparison DataFrame."""
    df = pd.DataFrame(results).T
    
    # Reorder columns
    cols = ['MRR', 'Hits@1', 'Hits@3', 'Hits@10', 'Mean Rank']
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    
    return df


def plot_metrics_comparison(results, save_path):
    """Create bar chart comparing metrics across models."""
    df = create_comparison_table(results)
    
    # Select metrics for comparison (exclude Mean Rank as it has different scale)
    metrics = ['MRR', 'Hits@1', 'Hits@3', 'Hits@10']
    metrics = [m for m in metrics if m in df.columns]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics))
    width = 0.25
    multiplier = 0
    
    colors = {'TransE': '#3498db', 'ComplEx': '#e74c3c', 'TriModel': '#2ecc71'}
    
    for model_name in df.index:
        values = [df.loc[model_name, m] if m in df.columns else 0 for m in metrics]
        offset = width * multiplier
        bars = ax.bar(x + offset, values, width, label=model_name, color=colors.get(model_name, 'gray'))
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
        
        multiplier += 1
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_title('Knowledge Graph Embedding Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_radar_chart(results, save_path):
    """Create radar/spider chart for model comparison."""
    df = create_comparison_table(results)
    
    metrics = ['MRR', 'Hits@1', 'Hits@3', 'Hits@10']
    metrics = [m for m in metrics if m in df.columns]
    
    # Number of variables
    num_vars = len(metrics)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    colors = {'TransE': '#3498db', 'ComplEx': '#e74c3c', 'TriModel': '#2ecc71'}
    
    for model_name in df.index:
        values = [df.loc[model_name, m] for m in metrics]
        values += values[:1]  # Complete the loop
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors.get(model_name, 'gray'))
        ax.fill(angles, values, alpha=0.15, color=colors.get(model_name, 'gray'))
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('Model Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_training_curves(losses, save_path):
    """Plot training loss curves for all models."""
    if not losses:
        print("No training loss data available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'TransE': '#3498db', 'ComplEx': '#e74c3c', 'TriModel': '#2ecc71'}
    
    for model_name, df in losses.items():
        if 'epoch' in df.columns and 'loss' in df.columns:
            final_loss = df['loss'].iloc[-1]
            ax.plot(df['epoch'], df['loss'], 
                   label=f'{model_name} (final: {final_loss:.4f})', 
                   color=colors.get(model_name, 'gray'), linewidth=2)
            # Annotate final loss value on the curve
            ax.annotate(f'{final_loss:.4f}',
                       xy=(df['epoch'].iloc[-1], final_loss),
                       xytext=(5, 0), textcoords='offset points',
                       fontsize=9, color=colors.get(model_name, 'gray'),
                       fontweight='bold', va='center')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_improvement_chart(results, save_path):
    """Plot percentage improvement over baseline (TransE)."""
    df = create_comparison_table(results)
    
    if 'TransE' not in df.index:
        print("TransE results not found for improvement comparison")
        return
    
    metrics = ['MRR', 'Hits@1', 'Hits@3', 'Hits@10']
    metrics = [m for m in metrics if m in df.columns]
    
    baseline = df.loc['TransE']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    colors = {'ComplEx': '#e74c3c', 'TriModel': '#2ecc71'}
    
    multiplier = 0
    for model_name in ['ComplEx', 'TriModel']:
        if model_name in df.index:
            improvements = []
            for m in metrics:
                if baseline[m] > 0:
                    pct = ((df.loc[model_name, m] - baseline[m]) / baseline[m]) * 100
                else:
                    pct = 0
                improvements.append(pct)
            
            offset = width * multiplier
            bars = ax.bar(x + offset, improvements, width, label=f'{model_name} vs TransE', 
                         color=colors.get(model_name, 'gray'))
            
            # Add value labels
            for bar, val in zip(bars, improvements):
                height = bar.get_height()
                ax.annotate(f'{val:+.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
            
            multiplier += 1
    
    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_title('Performance Improvement over TransE (Baseline)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=10)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def generate_latex_table(results):
    """Generate LaTeX table for academic papers."""
    df = create_comparison_table(results)
    
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\caption{Knowledge Graph Embedding Model Comparison}\n"
    latex += "\\begin{tabular}{l" + "c" * len(df.columns) + "}\n"
    latex += "\\hline\n"
    latex += "Model & " + " & ".join(df.columns) + " \\\\\n"
    latex += "\\hline\n"
    
    # Find best values for each column
    best_vals = df.max() if 'Mean Rank' not in df.columns else None
    
    for model in df.index:
        row = [model]
        for col in df.columns:
            val = df.loc[model, col]
            # Bold the best value (highest for MRR/Hits, lowest for Mean Rank)
            if col == 'Mean Rank':
                is_best = val == df[col].min()
            else:
                is_best = val == df[col].max()
            
            if is_best:
                row.append(f"\\textbf{{{val:.4f}}}")
            else:
                row.append(f"{val:.4f}")
        
        latex += " & ".join(row) + " \\\\\n"
    
    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\label{tab:model_comparison}\n"
    latex += "\\end{table}"
    
    return latex


def main():
    print("=" * 60)
    print("KNOWLEDGE GRAPH EMBEDDING MODEL COMPARISON")
    print("=" * 60)
    
    # Load results
    results = load_all_results()
    losses = load_training_losses()
    model_sizes = get_model_sizes()
    
    # Create comparison table
    print("\n📊 Performance Comparison Table:")
    print("-" * 60)
    df = create_comparison_table(results)
    print(df.to_string())
    
    # Calculate improvements
    if 'TransE' in df.index:
        print("\n📈 Improvement over TransE (baseline):")
        print("-" * 60)
        baseline = df.loc['TransE']
        for model in ['ComplEx', 'TriModel']:
            if model in df.index:
                print(f"\n{model}:")
                for col in df.columns:
                    if col != 'Mean Rank' and baseline[col] > 0:
                        pct = ((df.loc[model, col] - baseline[col]) / baseline[col]) * 100
                        print(f"  {col}: {pct:+.2f}%")
    
    # Model sizes
    if model_sizes:
        print("\n💾 Model Sizes:")
        print("-" * 60)
        for model, size in model_sizes.items():
            print(f"  {model}: {size:.2f} MB")
    
    # Generate plots
    print("\n🎨 Generating visualizations...")
    print("-" * 60)
    
    plot_metrics_comparison(results, os.path.join(OUTPUT_DIR, "figures", "metrics_comparison.png"))
    plot_radar_chart(results, os.path.join(OUTPUT_DIR, "figures", "radar_chart.png"))
    plot_training_curves(losses, os.path.join(OUTPUT_DIR, "figures", "training_curves.png"))
    plot_improvement_chart(results, os.path.join(OUTPUT_DIR, "figures", "improvement_chart.png"))
    
    # Save comparison table
    df.to_csv(os.path.join(OUTPUT_DIR, "comparison_table.csv"))
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'comparison_table.csv')}")
    
    # Save LaTeX table
    latex_table = generate_latex_table(results)
    with open(os.path.join(OUTPUT_DIR, "latex_table.tex"), 'w') as f:
        f.write(latex_table)
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'latex_table.tex')}")
    
    # Save summary report
    with open(os.path.join(OUTPUT_DIR, "comparison_report.txt"), 'w') as f:
        f.write("KNOWLEDGE GRAPH EMBEDDING MODEL COMPARISON\n")
        f.write("=" * 60 + "\n\n")
        f.write("Performance Metrics (Filtered Ranking)\n")
        f.write("-" * 60 + "\n")
        f.write(df.to_string())
        f.write("\n\n")
        
        # Find best model for each metric
        f.write("Best Model per Metric:\n")
        f.write("-" * 60 + "\n")
        for col in df.columns:
            if col == 'Mean Rank':
                best = df[col].idxmin()
            else:
                best = df[col].idxmax()
            f.write(f"  {col}: {best} ({df.loc[best, col]:.4f})\n")
        
        f.write("\n\nConclusion:\n")
        f.write("-" * 60 + "\n")
        
        # Determine overall best
        mrr_best = df['MRR'].idxmax() if 'MRR' in df.columns else None
        if mrr_best:
            f.write(f"Based on MRR (the most commonly used metric), {mrr_best} achieves the best performance.\n")
    
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'comparison_report.txt')}")
    
    print("\n" + "=" * 60)
    print("✅ Comparison complete! Check 'outputs_comparison' folder.")
    print("=" * 60)


if __name__ == "__main__":
    main()
