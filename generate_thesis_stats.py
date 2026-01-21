"""
Generate Dataset Statistics and Summary Tables for Thesis
==========================================================
"""

import pandas as pd
import os

# Load data from all splits
train = pd.read_csv('data/complex/train.txt', sep='\t', header=None, names=['source', 'relation', 'target'])
valid = pd.read_csv('data/complex/valid.txt', sep='\t', header=None, names=['source', 'relation', 'target'])
test = pd.read_csv('data/complex/test.txt', sep='\t', header=None, names=['source', 'relation', 'target'])

all_df = pd.concat([train, valid, test])

# Basic statistics
print('=' * 70)
print('DATASET STATISTICS FOR THESIS')
print('=' * 70)

print('\n### Table 1: Dataset Split Statistics ###')
print(f'| Split      | Triples |')
print(f'|------------|---------|')
print(f'| Training   | {len(train):,} |')
print(f'| Validation | {len(valid):,} |')
print(f'| Test       | {len(test):,} |')
print(f'| **Total**  | **{len(all_df):,}** |')

# Entity statistics
all_entities = set(all_df['source']) | set(all_df['target'])
drugs = {e for e in all_entities if str(e).startswith('DB')}
dti_df = all_df[all_df['relation'] == 'DRUG_TARGET']
proteins = set(dti_df['target'])
other_entities = all_entities - drugs - proteins

print(f'\n### Table 2: Entity Statistics ###')
print(f'| Entity Type | Count |')
print(f'|-------------|-------|')
print(f'| Drugs (DrugBank IDs) | {len(drugs):,} |')
print(f'| Proteins (UniProt IDs) | {len(proteins):,} |')
print(f'| Other Entities | {len(other_entities):,} |')
print(f'| **Total Entities** | **{len(all_entities):,}** |')

# Relation statistics
relations = all_df['relation'].unique()
print(f'\n### Table 3: Relation Statistics ###')
print(f'| Relation | Train | Valid | Test | Total |')
print(f'|----------|-------|-------|------|-------|')
for rel in sorted(relations):
    tr = len(train[train['relation'] == rel])
    va = len(valid[valid['relation'] == rel])
    te = len(test[test['relation'] == rel])
    print(f'| {rel} | {tr:,} | {va:,} | {te:,} | {tr+va+te:,} |')

# DTI specific
dti_train = train[train['relation'] == 'DRUG_TARGET']
dti_valid = valid[valid['relation'] == 'DRUG_TARGET']
dti_test = test[test['relation'] == 'DRUG_TARGET']

unique_drugs_dti = len(set(dti_df['source']))
unique_targets_dti = len(set(dti_df['target']))

print(f'\n### Table 4: Drug-Target Interaction (DTI) Statistics ###')
print(f'| Metric | Value |')
print(f'|--------|-------|')
print(f'| DTI pairs (Train) | {len(dti_train):,} |')
print(f'| DTI pairs (Valid) | {len(dti_valid):,} |')
print(f'| DTI pairs (Test) | {len(dti_test):,} |')
print(f'| DTI pairs (Total) | {len(dti_df):,} |')
print(f'| Unique drugs in DTI | {unique_drugs_dti:,} |')
print(f'| Unique targets in DTI | {unique_targets_dti:,} |')

# Sparsity
possible_dti = len(drugs) * len(proteins)
actual_dti = len(dti_df)
sparsity = 1 - (actual_dti / possible_dti) if possible_dti > 0 else 0
print(f'| Possible DTI pairs | {possible_dti:,} |')
print(f'| DTI Sparsity | {sparsity*100:.4f}% |')

# Evaluation results summary
print('\n### Table 5: DTI Prediction Results (Degree-Matched Sampling, NEG_RATIO=10) ###')
print(f'| Model | AUC-ROC | AUC-PR | Best F1 | Precision | Recall |')
print(f'|-------|---------|--------|---------|-----------|--------|')
print(f'| TransE | 0.6095 | 0.1738 | 0.2263 | 0.1893 | 0.2813 |')
print(f'| ComplEx | 0.7778 | 0.4040 | 0.4361 | 0.4297 | 0.4426 |')
print(f'| TriModel | 0.7642 | 0.3429 | 0.3852 | 0.3505 | 0.4274 |')

# Random baseline
pos_ratio = len(dti_test) / (len(dti_test) + len(dti_test) * 10)  # With NEG_RATIO=10
print(f'\n### Table 6: Baseline Comparison ###')
print(f'| Model | AUC-ROC | AUC-PR |')
print(f'|-------|---------|--------|')
print(f'| Random Baseline | 0.500 | {pos_ratio:.4f} |')
print(f'| TransE | 0.6095 (+0.110) | 0.1738 (+{0.1738-pos_ratio:.4f}) |')
print(f'| ComplEx | 0.7778 (+0.278) | 0.4040 (+{0.4040-pos_ratio:.4f}) |')
print(f'| TriModel | 0.7642 (+0.264) | 0.3429 (+{0.3429-pos_ratio:.4f}) |')

# Model comparison
print('\n### Key Findings ###')
print('1. ComplEx achieves the best overall performance (AUC-ROC: 0.778, AUC-PR: 0.404)')
print('2. TriModel performs comparably to ComplEx (AUC-ROC: 0.764, AUC-PR: 0.343)')
print('3. TransE underperforms significantly (AUC-ROC: 0.610, AUC-PR: 0.174)')
print('4. All models significantly outperform random baseline')
print(f'5. Dataset sparsity is {sparsity*100:.2f}%, indicating highly imbalanced DTI prediction task')

print('\n' + '=' * 70)
print('LaTeX TABLE FORMAT')
print('=' * 70)

print('''
\\begin{table}[h]
\\centering
\\caption{DTI Prediction Performance Comparison}
\\label{tab:dti_results}
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Model} & \\textbf{AUC-ROC} & \\textbf{AUC-PR} & \\textbf{Best F1} & \\textbf{Precision/Recall} \\\\
\\midrule
Random Baseline & 0.500 & 0.083 & - & - \\\\
TransE & 0.610 & 0.174 & 0.226 & 0.189/0.281 \\\\
ComplEx & \\textbf{0.778} & \\textbf{0.404} & \\textbf{0.436} & 0.430/0.443 \\\\
TriModel & 0.764 & 0.343 & 0.385 & 0.351/0.427 \\\\
\\bottomrule
\\end{tabular}
\\end{table}
''')

print('''
\\begin{table}[h]
\\centering
\\caption{Dataset Statistics}
\\label{tab:dataset}
\\begin{tabular}{lr}
\\toprule
\\textbf{Statistic} & \\textbf{Value} \\\\
\\midrule
Total Triples & ''' + f'{len(all_df):,}' + ''' \\\\
Total Entities & ''' + f'{len(all_entities):,}' + ''' \\\\
Total Relations & ''' + f'{len(relations)}' + ''' \\\\
\\midrule
Drugs (DrugBank) & ''' + f'{len(drugs):,}' + ''' \\\\
Proteins (UniProt) & ''' + f'{len(proteins):,}' + ''' \\\\
DTI Pairs & ''' + f'{len(dti_df):,}' + ''' \\\\
DTI Sparsity & ''' + f'{sparsity*100:.2f}\\%' + ''' \\\\
\\bottomrule
\\end{tabular}
\\end{table}
''')

# Save to file
output_path = 'outputs_dti_evaluation_fixed/thesis_statistics.txt'
os.makedirs('outputs_dti_evaluation_fixed', exist_ok=True)

with open(output_path, 'w') as f:
    f.write('DATASET STATISTICS FOR THESIS\n')
    f.write('=' * 70 + '\n\n')
    f.write(f'Total Triples: {len(all_df):,}\n')
    f.write(f'  - Training: {len(train):,}\n')
    f.write(f'  - Validation: {len(valid):,}\n')
    f.write(f'  - Test: {len(test):,}\n\n')
    f.write(f'Total Entities: {len(all_entities):,}\n')
    f.write(f'  - Drugs: {len(drugs):,}\n')
    f.write(f'  - Proteins: {len(proteins):,}\n')
    f.write(f'  - Other: {len(other_entities):,}\n\n')
    f.write(f'Total Relations: {len(relations)}\n\n')
    f.write(f'DTI Statistics:\n')
    f.write(f'  - Total DTI pairs: {len(dti_df):,}\n')
    f.write(f'  - Unique drugs: {unique_drugs_dti:,}\n')
    f.write(f'  - Unique targets: {unique_targets_dti:,}\n')
    f.write(f'  - Sparsity: {sparsity*100:.4f}%\n')

print(f'\nStatistics saved to: {output_path}')
