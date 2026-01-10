"""
Fact Prediction GUI
====================
A graphical interface for predicting drug-target interactions
and scoring knowledge graph facts.

Usage:
    python predict_fact_gui.py
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import torch
import threading

from model import TransE, ComplEx, TriModel


class PredictFactGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("KGE Fact Prediction Tool")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # Model variables
        self.model = None
        self.entity2id = None
        self.relation2id = None
        self.model_type = None
        self.device = "cpu"
        
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
        task_frame = ttk.LabelFrame(main_frame, text="2. Select Task", padding="10")
        task_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        
        self.task_var = tk.StringVar(value="score")
        tasks = [("Score a Fact", "score"),
                 ("Predict Tails (?)", "tail"),
                 ("Predict Heads (?)", "head")]
        
        for i, (name, value) in enumerate(tasks):
            rb = ttk.Radiobutton(task_frame, text=name, value=value, 
                                variable=self.task_var, command=self.update_input_fields)
            rb.grid(row=0, column=i, padx=20)
        
        # ===== Input Section =====
        input_frame = ttk.LabelFrame(main_frame, text="3. Enter Query", padding="10")
        input_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        input_frame.columnconfigure(1, weight=1)
        input_frame.columnconfigure(3, weight=1)
        input_frame.columnconfigure(5, weight=1)
        
        # Head entity
        self.head_label = ttk.Label(input_frame, text="Head Entity:")
        self.head_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.head_var = tk.StringVar()
        self.head_entry = ttk.Entry(input_frame, textvariable=self.head_var, width=25)
        self.head_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        # Relation
        ttk.Label(input_frame, text="Relation:").grid(row=0, column=2, padx=5, pady=5, sticky="e")
        self.relation_var = tk.StringVar()
        self.relation_combo = ttk.Combobox(input_frame, textvariable=self.relation_var, width=25)
        self.relation_combo.grid(row=0, column=3, padx=5, pady=5, sticky="ew")
        
        # Tail entity
        self.tail_label = ttk.Label(input_frame, text="Tail Entity:")
        self.tail_label.grid(row=0, column=4, padx=5, pady=5, sticky="e")
        self.tail_var = tk.StringVar()
        self.tail_entry = ttk.Entry(input_frame, textvariable=self.tail_var, width=25)
        self.tail_entry.grid(row=0, column=5, padx=5, pady=5, sticky="ew")
        
        # Top-K for predictions
        ttk.Label(input_frame, text="Top-K:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.topk_var = tk.StringVar(value="10")
        self.topk_spin = ttk.Spinbox(input_frame, from_=1, to=100, textvariable=self.topk_var, width=10)
        self.topk_spin.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # Example queries
        example_frame = ttk.Frame(input_frame)
        example_frame.grid(row=1, column=2, columnspan=4, sticky="w", padx=5)
        ttk.Label(example_frame, text="Examples:").pack(side="left", padx=5)
        ttk.Button(example_frame, text="Drug→Target", 
                  command=lambda: self.set_example("DB00001", "DRUG_TARGET", "")).pack(side="left", padx=2)
        ttk.Button(example_frame, text="Protein Interaction", 
                  command=lambda: self.set_example("P61026", "INTERACT_WITH", "Q9H0K6")).pack(side="left", padx=2)
        
        # ===== Run Button =====
        run_frame = ttk.Frame(main_frame)
        run_frame.grid(row=3, column=0, pady=10)
        
        self.run_btn = ttk.Button(run_frame, text="Run Prediction", command=self.run_prediction, 
                                  style="Accent.TButton")
        self.run_btn.pack(side="left", padx=5)
        
        self.clear_btn = ttk.Button(run_frame, text="Clear Results", command=self.clear_results)
        self.clear_btn.pack(side="left", padx=5)
        
        # ===== Results Section =====
        results_frame = ttk.LabelFrame(main_frame, text="4. Results", padding="10")
        results_frame.grid(row=4, column=0, sticky="nsew", pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, width=80, 
                                                       font=("Consolas", 10))
        self.results_text.grid(row=0, column=0, sticky="nsew")
        
        # ===== Status Bar =====
        self.status_var = tk.StringVar(value="Ready. Please load a model first.")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief="sunken", anchor="w")
        status_bar.grid(row=5, column=0, sticky="ew")
        
        # Configure styles
        style = ttk.Style()
        style.configure("Accent.TButton", font=("Arial", 10, "bold"))
    
    def set_example(self, head, relation, tail):
        """Set example values in the input fields."""
        self.head_var.set(head)
        self.relation_var.set(relation)
        self.tail_var.set(tail)
        if not tail:  # If no tail, switch to tail prediction mode
            self.task_var.set("tail")
            self.update_input_fields()
    
    def update_input_fields(self):
        """Update input field states based on selected task."""
        task = self.task_var.get()
        
        if task == "score":
            self.head_entry.config(state="normal")
            self.tail_entry.config(state="normal")
            self.head_label.config(text="Head Entity:")
            self.tail_label.config(text="Tail Entity:")
        elif task == "tail":
            self.head_entry.config(state="normal")
            self.tail_entry.config(state="disabled")
            self.tail_var.set("")
            self.head_label.config(text="Head Entity:")
            self.tail_label.config(text="Tail Entity: (predicted)")
        elif task == "head":
            self.head_entry.config(state="disabled")
            self.tail_entry.config(state="normal")
            self.head_var.set("")
            self.head_label.config(text="Head Entity: (predicted)")
            self.tail_label.config(text="Tail Entity:")
    
    def load_model(self):
        """Load the selected model."""
        model_name = self.model_var.get()
        
        model_paths = {
            "TransE": "outputs_transe/transe_model.pt",
            "ComplEx": "outputs_complex/complex_model.pt",
            "TriModel": "outputs_trimodel/trimodel_model.pt"
        }
        
        path = model_paths.get(model_name)
        if not os.path.exists(path):
            messagebox.showerror("Error", f"Model file not found: {path}")
            return
        
        self.status_var.set(f"Loading {model_name} model...")
        self.root.update()
        
        try:
            ckpt = torch.load(path, map_location=self.device)
            
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
            
            # Update relation dropdown
            self.relation_combo['values'] = list(self.relation2id.keys())
            
            self.model_status.config(text=f"✓ {model_name} loaded", foreground="green")
            self.status_var.set(f"{model_name} loaded: {len(self.entity2id):,} entities, "
                               f"{len(self.relation2id)} relations")
            
            # Show info in results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Model: {model_name}\n")
            self.results_text.insert(tk.END, f"Entities: {len(self.entity2id):,}\n")
            self.results_text.insert(tk.END, f"Relations: {len(self.relation2id)}\n")
            self.results_text.insert(tk.END, f"\nAvailable relations:\n")
            for rel in self.relation2id.keys():
                self.results_text.insert(tk.END, f"  - {rel}\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.status_var.set("Error loading model")
    
    def run_prediction(self):
        """Run the selected prediction task."""
        if self.model is None:
            messagebox.showwarning("Warning", "Please load a model first!")
            return
        
        task = self.task_var.get()
        
        try:
            if task == "score":
                self.score_fact()
            elif task == "tail":
                self.predict_tails()
            elif task == "head":
                self.predict_heads()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set(f"Error: {str(e)}")
    
    @torch.no_grad()
    def score_fact(self):
        """Score a single fact."""
        head = self.head_var.get().strip()
        relation = self.relation_var.get().strip()
        tail = self.tail_var.get().strip()
        
        if not head or not relation or not tail:
            messagebox.showwarning("Warning", "Please enter head, relation, and tail!")
            return
        
        # Validate inputs
        if head not in self.entity2id:
            messagebox.showerror("Error", f"Unknown head entity: {head}")
            return
        if relation not in self.relation2id:
            messagebox.showerror("Error", f"Unknown relation: {relation}")
            return
        if tail not in self.entity2id:
            messagebox.showerror("Error", f"Unknown tail entity: {tail}")
            return
        
        # Encode and score
        h_id = self.entity2id[head]
        r_id = self.relation2id[relation]
        t_id = self.entity2id[tail]
        
        triple = torch.tensor([[h_id, r_id, t_id]], dtype=torch.long, device=self.device)
        score = self.model(triple).item()
        
        # Display results
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "=" * 60 + "\n")
        self.results_text.insert(tk.END, "FACT SCORING RESULT\n")
        self.results_text.insert(tk.END, "=" * 60 + "\n\n")
        self.results_text.insert(tk.END, f"Triple: ({head}, {relation}, {tail})\n\n")
        self.results_text.insert(tk.END, f"Score: {score:.6f}\n\n")
        
        if self.model_type == "TransE":
            self.results_text.insert(tk.END, "(TransE: lower score = more plausible)\n")
        else:
            self.results_text.insert(tk.END, f"({self.model_type}: higher score = more plausible)\n")
        
        self.status_var.set(f"Scored fact: {score:.4f}")
    
    @torch.no_grad()
    def predict_tails(self):
        """Predict top-k tail entities."""
        head = self.head_var.get().strip()
        relation = self.relation_var.get().strip()
        top_k = int(self.topk_var.get())
        
        if not head or not relation:
            messagebox.showwarning("Warning", "Please enter head and relation!")
            return
        
        if head not in self.entity2id:
            messagebox.showerror("Error", f"Unknown head entity: {head}")
            return
        if relation not in self.relation2id:
            messagebox.showerror("Error", f"Unknown relation: {relation}")
            return
        
        self.status_var.set("Computing predictions...")
        self.root.update()
        
        h_id = self.entity2id[head]
        r_id = self.relation2id[relation]
        id2entity = {v: k for k, v in self.entity2id.items()}
        num_entities = len(self.entity2id)
        
        # Score all tails
        all_tails = torch.arange(num_entities, device=self.device)
        h_ids = torch.full((num_entities,), h_id, dtype=torch.long, device=self.device)
        r_ids = torch.full((num_entities,), r_id, dtype=torch.long, device=self.device)
        
        triples = torch.stack([h_ids, r_ids, all_tails], dim=1)
        scores = self.model(triples).cpu().numpy()
        
        # Sort
        if self.model_type == "TransE":
            sorted_indices = scores.argsort()
        else:
            sorted_indices = (-scores).argsort()
        
        # Display results
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "=" * 60 + "\n")
        self.results_text.insert(tk.END, f"TOP-{top_k} TAIL PREDICTIONS\n")
        self.results_text.insert(tk.END, f"Query: ({head}, {relation}, ?)\n")
        self.results_text.insert(tk.END, "=" * 60 + "\n\n")
        
        if self.model_type == "TransE":
            self.results_text.insert(tk.END, "(TransE: lower score = more plausible)\n\n")
        else:
            self.results_text.insert(tk.END, f"({self.model_type}: higher score = more plausible)\n\n")
        
        self.results_text.insert(tk.END, f"{'Rank':<6} {'Entity':<40} {'Score':<15}\n")
        self.results_text.insert(tk.END, "-" * 60 + "\n")
        
        for rank, idx in enumerate(sorted_indices[:top_k], 1):
            entity = id2entity[idx]
            score = scores[idx]
            self.results_text.insert(tk.END, f"{rank:<6} {entity:<40} {score:<15.6f}\n")
        
        self.status_var.set(f"Found top-{top_k} tail predictions")
    
    @torch.no_grad()
    def predict_heads(self):
        """Predict top-k head entities."""
        relation = self.relation_var.get().strip()
        tail = self.tail_var.get().strip()
        top_k = int(self.topk_var.get())
        
        if not relation or not tail:
            messagebox.showwarning("Warning", "Please enter relation and tail!")
            return
        
        if tail not in self.entity2id:
            messagebox.showerror("Error", f"Unknown tail entity: {tail}")
            return
        if relation not in self.relation2id:
            messagebox.showerror("Error", f"Unknown relation: {relation}")
            return
        
        self.status_var.set("Computing predictions...")
        self.root.update()
        
        r_id = self.relation2id[relation]
        t_id = self.entity2id[tail]
        id2entity = {v: k for k, v in self.entity2id.items()}
        num_entities = len(self.entity2id)
        
        # Score all heads
        all_heads = torch.arange(num_entities, device=self.device)
        r_ids = torch.full((num_entities,), r_id, dtype=torch.long, device=self.device)
        t_ids = torch.full((num_entities,), t_id, dtype=torch.long, device=self.device)
        
        triples = torch.stack([all_heads, r_ids, t_ids], dim=1)
        scores = self.model(triples).cpu().numpy()
        
        # Sort
        if self.model_type == "TransE":
            sorted_indices = scores.argsort()
        else:
            sorted_indices = (-scores).argsort()
        
        # Display results
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "=" * 60 + "\n")
        self.results_text.insert(tk.END, f"TOP-{top_k} HEAD PREDICTIONS\n")
        self.results_text.insert(tk.END, f"Query: (?, {relation}, {tail})\n")
        self.results_text.insert(tk.END, "=" * 60 + "\n\n")
        
        if self.model_type == "TransE":
            self.results_text.insert(tk.END, "(TransE: lower score = more plausible)\n\n")
        else:
            self.results_text.insert(tk.END, f"({self.model_type}: higher score = more plausible)\n\n")
        
        self.results_text.insert(tk.END, f"{'Rank':<6} {'Entity':<40} {'Score':<15}\n")
        self.results_text.insert(tk.END, "-" * 60 + "\n")
        
        for rank, idx in enumerate(sorted_indices[:top_k], 1):
            entity = id2entity[idx]
            score = scores[idx]
            self.results_text.insert(tk.END, f"{rank:<6} {entity:<40} {score:<15.6f}\n")
        
        self.status_var.set(f"Found top-{top_k} head predictions")
    
    def clear_results(self):
        """Clear the results text area."""
        self.results_text.delete(1.0, tk.END)
        self.status_var.set("Results cleared")


def main():
    root = tk.Tk()
    app = PredictFactGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
