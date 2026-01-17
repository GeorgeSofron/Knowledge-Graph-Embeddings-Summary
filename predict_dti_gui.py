"""
Drug-Target Interaction Prediction GUI
========================================
A specialized graphical interface for predicting drug-target interactions
using knowledge graph embedding models.

Features:
- Drug and target autocomplete dropdowns
- Novel vs. Known (training) label indicator
- Confidence score conversion
- Batch prediction mode
- CSV export functionality

Usage:
    python predict_dti_gui.py
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import torch
import threading
import csv
from datetime import datetime

from model import TransE, ComplEx, TriModel


class DrugTargetPredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Drug-Target Interaction Prediction Tool")
        self.root.geometry("1000x800")
        self.root.resizable(True, True)
        
        # Model variables
        self.model = None
        self.entity2id = None
        self.relation2id = None
        self.model_type = None
        self.device = "cpu"
        
        # DTI-specific variables
        self.drugs = []  # List of drug IDs (DB*)
        self.targets = []  # List of target IDs (P*, Q*)
        self.training_pairs = set()  # Known (drug, target) pairs from training
        self.dti_relation_id = None  # ID of DRUG_TARGET relation
        
        # Results storage for export
        self.current_results = []
        
        # Create main layout
        self.create_widgets()
        
    def create_widgets(self):
        """Create all GUI widgets."""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        # ===== Model Selection Section =====
        model_frame = ttk.LabelFrame(main_frame, text="1. Select Model", padding="10")
        model_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        model_frame.columnconfigure(1, weight=1)
        
        self.model_var = tk.StringVar(value="TransE")
        models = [("TransE", "outputs_transe/transe_model.pt"),
                  ("ComplEx", "outputs_complex/complex_model.pt"),
                  ("TriModel", "outputs_trimodel/trimodel_model.pt")]
        
        for i, (name, path) in enumerate(models):
            rb = ttk.Radiobutton(model_frame, text=name, value=name, variable=self.model_var)
            rb.grid(row=0, column=i, padx=10)
        
        self.load_btn = ttk.Button(model_frame, text="Load Model", command=self.load_model)
        self.load_btn.grid(row=0, column=3, padx=20)
        
        self.model_status = ttk.Label(model_frame, text="No model loaded", foreground="red")
        self.model_status.grid(row=0, column=4, padx=10)
        
        # ===== Task Selection =====
        task_frame = ttk.LabelFrame(main_frame, text="2. Select Prediction Task", padding="10")
        task_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        
        self.task_var = tk.StringVar(value="targets")
        tasks = [("Score Drug-Target Pair", "score"),
                 ("Predict Targets for Drug", "targets"),
                 ("Predict Drugs for Target", "drugs"),
                 ("Batch: All Targets for Drug", "batch_targets"),
                 ("Batch: All Drugs for Target", "batch_drugs")]
        
        for i, (name, value) in enumerate(tasks):
            rb = ttk.Radiobutton(task_frame, text=name, value=value, 
                                variable=self.task_var, command=self.update_input_fields)
            rb.grid(row=0, column=i, padx=10)
        
        # ===== Input Section =====
        input_frame = ttk.LabelFrame(main_frame, text="3. Enter Query", padding="10")
        input_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        input_frame.columnconfigure(1, weight=1)
        input_frame.columnconfigure(3, weight=1)
        
        # Drug selection
        self.drug_label = ttk.Label(input_frame, text="Drug (DB ID):")
        self.drug_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.drug_var = tk.StringVar()
        self.drug_combo = ttk.Combobox(input_frame, textvariable=self.drug_var, width=30)
        self.drug_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.drug_combo.bind('<KeyRelease>', lambda e: self.filter_combobox(e, 'drug'))
        
        # Target selection
        self.target_label = ttk.Label(input_frame, text="Target Protein (UniProt ID):")
        self.target_label.grid(row=0, column=2, padx=5, pady=5, sticky="e")
        self.target_var = tk.StringVar()
        self.target_combo = ttk.Combobox(input_frame, textvariable=self.target_var, width=30)
        self.target_combo.grid(row=0, column=3, padx=5, pady=5, sticky="ew")
        self.target_combo.bind('<KeyRelease>', lambda e: self.filter_combobox(e, 'target'))
        
        # Options row
        options_frame = ttk.Frame(input_frame)
        options_frame.grid(row=1, column=0, columnspan=4, sticky="ew", pady=5)
        
        # Top-K for predictions
        ttk.Label(options_frame, text="Top-K Results:").pack(side="left", padx=5)
        self.topk_var = tk.StringVar(value="20")
        self.topk_spin = ttk.Spinbox(options_frame, from_=1, to=500, textvariable=self.topk_var, width=8)
        self.topk_spin.pack(side="left", padx=5)
        
        # Filter options
        ttk.Separator(options_frame, orient="vertical").pack(side="left", fill="y", padx=15)
        
        self.show_novel_only = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Show Novel Only (exclude training pairs)", 
                       variable=self.show_novel_only).pack(side="left", padx=5)
        
        self.show_known_only = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Show Known Only (training pairs)", 
                       variable=self.show_known_only).pack(side="left", padx=5)
        
        # Confidence threshold
        ttk.Separator(options_frame, orient="vertical").pack(side="left", fill="y", padx=15)
        ttk.Label(options_frame, text="Min Confidence %:").pack(side="left", padx=5)
        self.min_conf_var = tk.StringVar(value="0")
        self.min_conf_spin = ttk.Spinbox(options_frame, from_=0, to=100, 
                                         textvariable=self.min_conf_var, width=6)
        self.min_conf_spin.pack(side="left", padx=5)
        
        # ===== Run Button =====
        run_frame = ttk.Frame(main_frame)
        run_frame.grid(row=3, column=0, pady=10)
        
        self.run_btn = ttk.Button(run_frame, text="🔍 Run Prediction", 
                                  command=self.run_prediction_threaded)
        self.run_btn.pack(side="left", padx=5)
        
        self.export_btn = ttk.Button(run_frame, text="📁 Export to CSV", 
                                     command=self.export_results)
        self.export_btn.pack(side="left", padx=5)
        
        self.clear_btn = ttk.Button(run_frame, text="🗑️ Clear Results", command=self.clear_results)
        self.clear_btn.pack(side="left", padx=5)
        
        # ===== Results Section =====
        results_frame = ttk.LabelFrame(main_frame, text="4. Prediction Results", padding="10")
        results_frame.grid(row=4, column=0, sticky="nsew", pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # Create Treeview for results
        columns = ("Rank", "Drug", "Target", "Score", "Confidence", "Status")
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show="headings", height=15)
        
        for col in columns:
            self.results_tree.heading(col, text=col)
            if col == "Rank":
                self.results_tree.column(col, width=50, anchor="center")
            elif col in ("Score", "Confidence"):
                self.results_tree.column(col, width=100, anchor="center")
            elif col == "Status":
                self.results_tree.column(col, width=100, anchor="center")
            else:
                self.results_tree.column(col, width=150, anchor="w")
        
        # Scrollbars
        v_scroll = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_tree.yview)
        h_scroll = ttk.Scrollbar(results_frame, orient="horizontal", command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        self.results_tree.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")
        
        # Tag for coloring rows
        self.results_tree.tag_configure("novel", background="#e6ffe6")  # Light green
        self.results_tree.tag_configure("known", background="#fff3e6")  # Light orange
        
        # Summary text
        self.summary_text = scrolledtext.ScrolledText(results_frame, height=5, width=80, 
                                                       font=("Consolas", 9))
        self.summary_text.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        
        # ===== Status Bar =====
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=5, column=0, sticky="ew")
        status_frame.columnconfigure(0, weight=1)
        
        self.status_var = tk.StringVar(value="Ready. Please load a model first.")
        status_bar = ttk.Label(status_frame, textvariable=self.status_var, relief="sunken", anchor="w")
        status_bar.grid(row=0, column=0, sticky="ew")
        
        # Progress bar
        self.progress = ttk.Progressbar(status_frame, mode="indeterminate", length=150)
        self.progress.grid(row=0, column=1, padx=10)
    
    def filter_combobox(self, event, combo_type):
        """Filter combobox options based on typed text."""
        if combo_type == 'drug':
            combo = self.drug_combo
            items = self.drugs
        else:
            combo = self.target_combo
            items = self.targets
        
        typed = combo.get().upper()
        if typed == '':
            combo['values'] = items[:100]  # Show first 100 when empty
        else:
            filtered = [item for item in items if typed in item.upper()][:50]
            combo['values'] = filtered
    
    def update_input_fields(self):
        """Update input field states based on selected task."""
        task = self.task_var.get()
        
        if task == "score":
            self.drug_combo.config(state="normal")
            self.target_combo.config(state="normal")
            self.drug_label.config(text="Drug (DB ID):")
            self.target_label.config(text="Target Protein (UniProt ID):")
        elif task in ("targets", "batch_targets"):
            self.drug_combo.config(state="normal")
            self.target_combo.config(state="disabled")
            self.target_var.set("")
            self.drug_label.config(text="Drug (DB ID):")
            self.target_label.config(text="Target: (predicted)")
        elif task in ("drugs", "batch_drugs"):
            self.drug_combo.config(state="disabled")
            self.target_combo.config(state="normal")
            self.drug_var.set("")
            self.drug_label.config(text="Drug: (predicted)")
            self.target_label.config(text="Target Protein (UniProt ID):")
    
    def load_model(self):
        """Load the selected model and extract DTI-specific data."""
        model_name = self.model_var.get()
        
        model_paths = {
            "TransE": ("outputs_transe/transe_model.pt", "data/transe/train.txt"),
            "ComplEx": ("outputs_complex/complex_model.pt", "data/complex/train.txt"),
            "TriModel": ("outputs_trimodel/trimodel_model.pt", "data/trimodel/train.txt")
        }
        
        model_path, train_path = model_paths.get(model_name)
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"Model file not found: {model_path}")
            return
        
        self.status_var.set(f"Loading {model_name} model...")
        self.progress.start()
        self.root.update()
        
        try:
            ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Detect and load model
            if model_name == "TransE":
                self.model = TransE(
                    num_entities=ckpt["num_entities"],
                    num_relations=ckpt["num_relations"],
                    dim=ckpt["embedding_dim"],
                    p_norm=ckpt["p_norm"],
                ).to(self.device)
            elif model_name == "ComplEx":
                self.model = ComplEx(
                    num_entities=ckpt["num_entities"],
                    num_relations=ckpt["num_relations"],
                    dim=ckpt["embedding_dim"],
                    reg_weight=ckpt.get("reg_weight", 0.01),
                ).to(self.device)
            else:  # TriModel
                self.model = TriModel(
                    num_entities=ckpt["num_entities"],
                    num_relations=ckpt["num_relations"],
                    dim=ckpt["embedding_dim"],
                    reg_weight=ckpt.get("reg_weight", 0.01),
                ).to(self.device)
            
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()
            
            self.entity2id = ckpt["entity2id"]
            self.relation2id = ckpt["relation2id"]
            self.model_type = model_name
            
            # Get DRUG_TARGET relation ID
            if "DRUG_TARGET" in self.relation2id:
                self.dti_relation_id = self.relation2id["DRUG_TARGET"]
            else:
                messagebox.showwarning("Warning", "DRUG_TARGET relation not found in model!")
            
            # Separate drugs and targets
            self.drugs = sorted([e for e in self.entity2id.keys() if e.startswith("DB")])
            self.targets = sorted([e for e in self.entity2id.keys() 
                                   if e.startswith("P") or e.startswith("Q")])
            
            # Update dropdowns
            self.drug_combo['values'] = self.drugs[:100]
            self.target_combo['values'] = self.targets[:100]
            
            # Load training pairs for novel/known detection
            self.training_pairs = set()
            if os.path.exists(train_path):
                with open(train_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            head, relation, tail = parts[0], parts[1], parts[2]
                            if relation == "DRUG_TARGET":
                                self.training_pairs.add((head, tail))
            
            self.model_status.config(text=f"✓ {model_name} loaded", foreground="green")
            self.status_var.set(f"{model_name} loaded: {len(self.drugs):,} drugs, "
                               f"{len(self.targets):,} targets, "
                               f"{len(self.training_pairs):,} known DTI pairs")
            
            # Show summary
            self.summary_text.delete(1.0, tk.END)
            self.summary_text.insert(tk.END, f"Model: {model_name}\n")
            self.summary_text.insert(tk.END, f"Total Entities: {len(self.entity2id):,}\n")
            self.summary_text.insert(tk.END, f"  - Drugs (DB*): {len(self.drugs):,}\n")
            self.summary_text.insert(tk.END, f"  - Targets (P*/Q*): {len(self.targets):,}\n")
            self.summary_text.insert(tk.END, f"Known Drug-Target Pairs (Training): {len(self.training_pairs):,}\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.status_var.set("Error loading model")
        finally:
            self.progress.stop()
    
    def run_prediction_threaded(self):
        """Run prediction in a background thread."""
        if self.model is None:
            messagebox.showwarning("Warning", "Please load a model first!")
            return
        
        self.progress.start()
        self.run_btn.config(state="disabled")
        
        thread = threading.Thread(target=self.run_prediction)
        thread.start()
    
    def run_prediction(self):
        """Run the selected prediction task."""
        try:
            task = self.task_var.get()
            
            if task == "score":
                self.score_dti()
            elif task in ("targets", "batch_targets"):
                self.predict_targets()
            elif task in ("drugs", "batch_drugs"):
                self.predict_drugs()
                
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.root.after(0, lambda: self.status_var.set(f"Error: {str(e)}"))
        finally:
            self.root.after(0, self.progress.stop)
            self.root.after(0, lambda: self.run_btn.config(state="normal"))
    
    def compute_confidence(self, scores):
        """Convert raw scores to confidence percentages (0-100)."""
        if self.model_type == "TransE":
            # TransE: lower is better, so invert
            # Use sigmoid-like transformation
            confidences = 100 * (1 / (1 + scores))
        else:
            # ComplEx/TriModel: higher is better
            # Normalize to 0-100 using sigmoid
            confidences = 100 / (1 + torch.exp(-torch.tensor(scores)))
            confidences = confidences.numpy()
        return confidences
    
    def is_novel(self, drug, target):
        """Check if a drug-target pair is novel (not in training data)."""
        return (drug, target) not in self.training_pairs
    
    @torch.no_grad()
    def score_dti(self):
        """Score a single drug-target pair."""
        drug = self.drug_var.get().strip()
        target = self.target_var.get().strip()
        
        if not drug or not target:
            self.root.after(0, lambda: messagebox.showwarning("Warning", 
                                                               "Please enter drug and target!"))
            return
        
        # Validate inputs
        if drug not in self.entity2id:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Unknown drug: {drug}"))
            return
        if target not in self.entity2id:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Unknown target: {target}"))
            return
        
        # Encode and score
        h_id = self.entity2id[drug]
        r_id = self.dti_relation_id
        t_id = self.entity2id[target]
        
        triple = torch.tensor([[h_id, r_id, t_id]], dtype=torch.long, device=self.device)
        score = self.model(triple).item()
        confidence = self.compute_confidence([score])[0]
        is_novel = self.is_novel(drug, target)
        status = "🆕 NOVEL" if is_novel else "📚 KNOWN"
        
        # Store result
        self.current_results = [(1, drug, target, score, confidence, status)]
        
        # Update UI
        def update_ui():
            # Clear and populate tree
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)
            
            tag = "novel" if is_novel else "known"
            self.results_tree.insert("", "end", values=(1, drug, target, f"{score:.6f}", 
                                                        f"{confidence:.1f}%", status), tags=(tag,))
            
            # Update summary
            self.summary_text.delete(1.0, tk.END)
            self.summary_text.insert(tk.END, f"Drug-Target Pair Scoring Result\n")
            self.summary_text.insert(tk.END, "=" * 50 + "\n")
            self.summary_text.insert(tk.END, f"Drug: {drug}\n")
            self.summary_text.insert(tk.END, f"Target: {target}\n")
            self.summary_text.insert(tk.END, f"Raw Score: {score:.6f}\n")
            self.summary_text.insert(tk.END, f"Confidence: {confidence:.1f}%\n")
            self.summary_text.insert(tk.END, f"Status: {status}\n")
            if is_novel:
                self.summary_text.insert(tk.END, "\n⚡ This is a NOVEL prediction (not in training data)!\n")
            else:
                self.summary_text.insert(tk.END, "\n📚 This pair was seen during training.\n")
            
            self.status_var.set(f"Scored: {drug} → {target} | Confidence: {confidence:.1f}%")
        
        self.root.after(0, update_ui)
    
    @torch.no_grad()
    def predict_targets(self):
        """Predict top-k target proteins for a drug."""
        drug = self.drug_var.get().strip()
        top_k = int(self.topk_var.get())
        min_conf = float(self.min_conf_var.get())
        
        if not drug:
            self.root.after(0, lambda: messagebox.showwarning("Warning", "Please enter a drug!"))
            return
        
        if drug not in self.entity2id:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Unknown drug: {drug}"))
            return
        
        self.root.after(0, lambda: self.status_var.set("Computing target predictions..."))
        
        h_id = self.entity2id[drug]
        r_id = self.dti_relation_id
        
        # Get target IDs only
        target_ids = [self.entity2id[t] for t in self.targets if t in self.entity2id]
        id2entity = {v: k for k, v in self.entity2id.items()}
        
        # Score all targets
        target_tensor = torch.tensor(target_ids, dtype=torch.long, device=self.device)
        h_ids = torch.full((len(target_ids),), h_id, dtype=torch.long, device=self.device)
        r_ids = torch.full((len(target_ids),), r_id, dtype=torch.long, device=self.device)
        
        triples = torch.stack([h_ids, r_ids, target_tensor], dim=1)
        scores = self.model(triples).cpu().numpy()
        confidences = self.compute_confidence(scores)
        
        # Sort
        if self.model_type == "TransE":
            sorted_indices = scores.argsort()
        else:
            sorted_indices = (-scores).argsort()
        
        # Build results with filtering
        results = []
        for idx in sorted_indices:
            target = id2entity[target_ids[idx]]
            score = scores[idx]
            conf = confidences[idx]
            is_novel = self.is_novel(drug, target)
            
            # Apply filters
            if conf < min_conf:
                continue
            if self.show_novel_only.get() and not is_novel:
                continue
            if self.show_known_only.get() and is_novel:
                continue
            
            status = "🆕 NOVEL" if is_novel else "📚 KNOWN"
            results.append((len(results) + 1, drug, target, score, conf, status, is_novel))
            
            if len(results) >= top_k:
                break
        
        self.current_results = results
        
        # Count novel vs known
        novel_count = sum(1 for r in results if r[6])
        known_count = len(results) - novel_count
        
        # Update UI
        def update_ui():
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)
            
            for rank, d, t, score, conf, status, is_novel in results:
                tag = "novel" if is_novel else "known"
                self.results_tree.insert("", "end", 
                                        values=(rank, d, t, f"{score:.6f}", f"{conf:.1f}%", status),
                                        tags=(tag,))
            
            # Update summary
            self.summary_text.delete(1.0, tk.END)
            self.summary_text.insert(tk.END, f"Target Predictions for Drug: {drug}\n")
            self.summary_text.insert(tk.END, "=" * 50 + "\n")
            self.summary_text.insert(tk.END, f"Total Results: {len(results)}\n")
            self.summary_text.insert(tk.END, f"  🆕 Novel predictions: {novel_count}\n")
            self.summary_text.insert(tk.END, f"  📚 Known (training) pairs: {known_count}\n")
            
            self.status_var.set(f"Found {len(results)} targets for {drug} "
                               f"({novel_count} novel, {known_count} known)")
        
        self.root.after(0, update_ui)
    
    @torch.no_grad()
    def predict_drugs(self):
        """Predict top-k drugs for a target protein."""
        target = self.target_var.get().strip()
        top_k = int(self.topk_var.get())
        min_conf = float(self.min_conf_var.get())
        
        if not target:
            self.root.after(0, lambda: messagebox.showwarning("Warning", "Please enter a target!"))
            return
        
        if target not in self.entity2id:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Unknown target: {target}"))
            return
        
        self.root.after(0, lambda: self.status_var.set("Computing drug predictions..."))
        
        t_id = self.entity2id[target]
        r_id = self.dti_relation_id
        
        # Get drug IDs only
        drug_ids = [self.entity2id[d] for d in self.drugs if d in self.entity2id]
        id2entity = {v: k for k, v in self.entity2id.items()}
        
        # Score all drugs
        drug_tensor = torch.tensor(drug_ids, dtype=torch.long, device=self.device)
        t_ids = torch.full((len(drug_ids),), t_id, dtype=torch.long, device=self.device)
        r_ids = torch.full((len(drug_ids),), r_id, dtype=torch.long, device=self.device)
        
        triples = torch.stack([drug_tensor, r_ids, t_ids], dim=1)
        scores = self.model(triples).cpu().numpy()
        confidences = self.compute_confidence(scores)
        
        # Sort
        if self.model_type == "TransE":
            sorted_indices = scores.argsort()
        else:
            sorted_indices = (-scores).argsort()
        
        # Build results with filtering
        results = []
        for idx in sorted_indices:
            drug = id2entity[drug_ids[idx]]
            score = scores[idx]
            conf = confidences[idx]
            is_novel = self.is_novel(drug, target)
            
            # Apply filters
            if conf < min_conf:
                continue
            if self.show_novel_only.get() and not is_novel:
                continue
            if self.show_known_only.get() and is_novel:
                continue
            
            status = "🆕 NOVEL" if is_novel else "📚 KNOWN"
            results.append((len(results) + 1, drug, target, score, conf, status, is_novel))
            
            if len(results) >= top_k:
                break
        
        self.current_results = results
        
        # Count novel vs known
        novel_count = sum(1 for r in results if r[6])
        known_count = len(results) - novel_count
        
        # Update UI
        def update_ui():
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)
            
            for rank, d, t, score, conf, status, is_novel in results:
                tag = "novel" if is_novel else "known"
                self.results_tree.insert("", "end", 
                                        values=(rank, d, t, f"{score:.6f}", f"{conf:.1f}%", status),
                                        tags=(tag,))
            
            # Update summary
            self.summary_text.delete(1.0, tk.END)
            self.summary_text.insert(tk.END, f"Drug Predictions for Target: {target}\n")
            self.summary_text.insert(tk.END, "=" * 50 + "\n")
            self.summary_text.insert(tk.END, f"Total Results: {len(results)}\n")
            self.summary_text.insert(tk.END, f"  🆕 Novel predictions: {novel_count}\n")
            self.summary_text.insert(tk.END, f"  📚 Known (training) pairs: {known_count}\n")
            
            self.status_var.set(f"Found {len(results)} drugs for {target} "
                               f"({novel_count} novel, {known_count} known)")
        
        self.root.after(0, update_ui)
    
    def export_results(self):
        """Export current results to CSV file."""
        if not self.current_results:
            messagebox.showwarning("Warning", "No results to export!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"dti_predictions_{timestamp}.csv"
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=default_name
        )
        
        if filepath:
            try:
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Rank", "Drug", "Target", "Score", "Confidence_%", "Status", "Is_Novel"])
                    
                    for row in self.current_results:
                        # Handle both score and prediction results
                        if len(row) == 6:  # Score result (no is_novel flag separately)
                            rank, drug, target, score, conf, status = row
                            is_novel = "NOVEL" in status
                        else:  # Prediction result
                            rank, drug, target, score, conf, status, is_novel = row
                        
                        writer.writerow([rank, drug, target, f"{score:.6f}", f"{conf:.1f}", 
                                        status.replace("🆕 ", "").replace("📚 ", ""), is_novel])
                
                self.status_var.set(f"Exported {len(self.current_results)} results to {filepath}")
                messagebox.showinfo("Success", f"Results exported to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {str(e)}")
    
    def clear_results(self):
        """Clear the results."""
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self.summary_text.delete(1.0, tk.END)
        self.current_results = []
        self.status_var.set("Results cleared")


def main():
    root = tk.Tk()
    app = DrugTargetPredictionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
