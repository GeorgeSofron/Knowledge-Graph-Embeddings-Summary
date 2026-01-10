"""
Knowledge Graph Embedding Models
=================================
Single source of truth for KG embedding models.
Both training and evaluation scripts import from here.

Models:
    - TransE: Translating Embeddings (Bordes et al., 2013)
    - ComplEx: Complex Embeddings (Trouillon et al., 2016)
    - TriModel: Tri-vector Embeddings (Kamaleldin et al.)

Loss Functions (aligned with libkge):
    - pairwise_hinge_loss: Margin-based ranking loss
    - pairwise_logistic_loss: Softplus-based pairwise loss
    - pointwise_logistic_loss: Pointwise logistic loss with targets
    - pointwise_hinge_loss: Pointwise hinge loss with targets
    - pointwise_square_error_loss: Squared error loss
    - bce_loss: Binary cross-entropy loss

Reference:
    Loss functions based on libkge by Sameh Kamaleldin
    https://github.com/samehkamaleldin/libkge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Loss Functions (aligned with libkge/embedding/losses.py)
# =============================================================================

def pairwise_hinge_loss(
    pos_scores: torch.Tensor, 
    neg_scores: torch.Tensor, 
    margin: float = 1.0, 
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Pairwise hinge (margin ranking) loss.
    
    For distance-based models (TransE): lower pos_scores are better,
    so we use: max(0, margin + pos - neg)
    
    For score-based models: higher pos_scores are better,
    so we use: max(0, margin + neg - pos)
    
    Args:
        pos_scores: Scores for positive triples
        neg_scores: Scores for negative triples
        margin: Margin value (default: 1.0)
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Loss value
    """
    # For score-based models (higher is better): margin + neg - pos
    loss = F.relu(margin + neg_scores - pos_scores)
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def pairwise_hinge_loss_distance(
    pos_dist: torch.Tensor, 
    neg_dist: torch.Tensor, 
    margin: float = 1.0, 
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Pairwise hinge loss for distance-based models (TransE).
    
    Lower distances are better, so we use: max(0, margin + pos - neg)
    
    Args:
        pos_dist: Distances for positive triples (lower is better)
        neg_dist: Distances for negative triples
        margin: Margin value (default: 1.0)
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Loss value
    """
    loss = F.relu(margin + pos_dist - neg_dist)
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def pairwise_logistic_loss(
    pos_scores: torch.Tensor, 
    neg_scores: torch.Tensor, 
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Pairwise logistic (softplus) loss.
    
    loss = softplus(neg_scores - pos_scores)
    
    For score-based models: higher pos_scores are better.
    
    Args:
        pos_scores: Scores for positive triples
        neg_scores: Scores for negative triples
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Loss value
    """
    loss = F.softplus(neg_scores - pos_scores)
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def pointwise_logistic_loss(
    scores: torch.Tensor, 
    targets: torch.Tensor, 
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Pointwise logistic loss.
    
    loss = softplus(-targets * scores)
    
    Targets should be +1 for positive, -1 for negative.
    
    Args:
        scores: Model scores
        targets: Target labels (+1 or -1)
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Loss value
    """
    loss = F.softplus(-targets * scores)
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def pointwise_hinge_loss(
    scores: torch.Tensor, 
    targets: torch.Tensor, 
    margin: float = 1.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Pointwise hinge loss.
    
    loss = max(0, margin - targets * scores)
    
    Targets should be +1 for positive, -1 for negative.
    
    Args:
        scores: Model scores
        targets: Target labels (+1 or -1)
        margin: Margin value (default: 1.0)
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Loss value
    """
    loss = F.relu(margin - targets * scores)
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def pointwise_square_error_loss(
    scores: torch.Tensor, 
    targets: torch.Tensor, 
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Pointwise squared error loss.
    
    loss = (scores - targets)^2
    
    Targets should be +1 for positive, -1 for negative (or 0/1).
    
    Args:
        scores: Model scores
        targets: Target labels
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Loss value
    """
    loss = (scores - targets) ** 2
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def bce_loss(
    pos_scores: torch.Tensor, 
    neg_scores: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Binary cross-entropy loss with logits.
    
    For score-based models (ComplEx, TriModel): higher scores are better.
    Positive triples should have high scores (label=1), negatives low (label=0).
    
    Args:
        pos_scores: Scores for positive triples
        neg_scores: Scores for negative triples
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Loss value
    """
    pos_labels = torch.ones_like(pos_scores)
    neg_labels = torch.zeros_like(neg_scores)
    
    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([pos_labels, neg_labels])
    
    return F.binary_cross_entropy_with_logits(scores, labels, reduction=reduction)


def compute_kge_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    loss_type: str = 'pairwise_logistic',
    margin: float = 1.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Unified loss computation for KGE models.
    
    Args:
        pos_scores: Scores for positive triples
        neg_scores: Scores for negative triples
        loss_type: One of:
            - 'pairwise_hinge' or 'pr_hinge': Pairwise hinge loss
            - 'pairwise_logistic' or 'pr_log': Pairwise logistic loss
            - 'pointwise_hinge' or 'pt_hinge': Pointwise hinge loss
            - 'pointwise_logistic' or 'pt_log': Pointwise logistic loss
            - 'pointwise_square' or 'pt_se': Pointwise squared error
            - 'bce': Binary cross-entropy
        margin: Margin for hinge losses (default: 1.0)
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Loss value
    """
    if loss_type in ['pairwise_hinge', 'pr_hinge']:
        return pairwise_hinge_loss(pos_scores, neg_scores, margin, reduction)
    
    elif loss_type in ['pairwise_logistic', 'pr_log']:
        return pairwise_logistic_loss(pos_scores, neg_scores, reduction)
    
    elif loss_type in ['pointwise_hinge', 'pt_hinge']:
        pos_targets = torch.ones_like(pos_scores)
        neg_targets = -torch.ones_like(neg_scores)
        scores = torch.cat([pos_scores, neg_scores])
        targets = torch.cat([pos_targets, neg_targets])
        return pointwise_hinge_loss(scores, targets, margin, reduction)
    
    elif loss_type in ['pointwise_logistic', 'pt_log']:
        pos_targets = torch.ones_like(pos_scores)
        neg_targets = -torch.ones_like(neg_scores)
        scores = torch.cat([pos_scores, neg_scores])
        targets = torch.cat([pos_targets, neg_targets])
        return pointwise_logistic_loss(scores, targets, reduction)
    
    elif loss_type in ['pointwise_square', 'pt_se']:
        pos_targets = torch.ones_like(pos_scores)
        neg_targets = -torch.ones_like(neg_scores)
        scores = torch.cat([pos_scores, neg_scores])
        targets = torch.cat([pos_targets, neg_targets])
        return pointwise_square_error_loss(scores, targets, reduction)
    
    elif loss_type == 'bce':
        return bce_loss(pos_scores, neg_scores, reduction)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Options: "
                        "['pairwise_hinge', 'pairwise_logistic', 'pointwise_hinge', "
                        "'pointwise_logistic', 'pointwise_square', 'bce']")


# =============================================================================
# Model Classes
# =============================================================================

class TransE(nn.Module):
    """
    TransE: Translating Embeddings for Modeling Multi-relational Data
    
    Score function: ||h + r - t||_p (lower is better)
    
    Args:
        num_entities: Number of unique entities in the knowledge graph
        num_relations: Number of unique relation types
        dim: Embedding dimension (default: 100)
        p_norm: Norm to use for distance calculation (default: 1 for L1)
    """
    
    def __init__(self, num_entities: int, num_relations: int, dim: int = 100, p_norm: int = 1):
        super().__init__()
        self.dim = dim
        self.p_norm = p_norm

        self.ent = nn.Embedding(num_entities, dim)
        self.rel = nn.Embedding(num_relations, dim)

        # Xavier initialization for stable training
        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight)

    def score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute TransE score: ||h + r - t||_p
        
        Args:
            h: Head entity embeddings
            r: Relation embeddings
            t: Tail entity embeddings
            
        Returns:
            Distance scores (lower = more plausible triple)
        """
        return torch.norm(h + r - t, p=self.p_norm, dim=-1)

    def forward(self, triples: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of triples.
        
        Args:
            triples: Tensor of shape (batch_size, 3) with columns [head_id, relation_id, tail_id]
            
        Returns:
            Scores for each triple
        """
        h = self.ent(triples[:, 0])
        r = self.rel(triples[:, 1])
        t = self.ent(triples[:, 2])
        return self.score(h, r, t)


class ComplEx(nn.Module):
    """
    ComplEx: Complex Embeddings for Simple Link Prediction
    
    Uses complex-valued embeddings where each entity/relation has 
    real and imaginary components.
    
    Score function: Re(⟨h, r, conj(t)⟩) = Re(Σ h_i * r_i * conj(t_i))
    
    Higher scores indicate more plausible triples (unlike TransE where lower is better).
    
    Args:
        num_entities: Number of unique entities in the knowledge graph
        num_relations: Number of unique relation types
        dim: Embedding dimension for each of real/imaginary parts (total = 2*dim)
        reg_weight: L2 regularization weight
    """
    
    def __init__(self, num_entities: int, num_relations: int, dim: int = 100, reg_weight: float = 0.01):
        super().__init__()
        self.dim = dim
        self.reg_weight = reg_weight
        self.num_entities = num_entities
        self.num_relations = num_relations
        
        # Real and imaginary parts of entity embeddings
        self.ent_re = nn.Embedding(num_entities, dim)
        self.ent_im = nn.Embedding(num_entities, dim)
        
        # Real and imaginary parts of relation embeddings
        self.rel_re = nn.Embedding(num_relations, dim)
        self.rel_im = nn.Embedding(num_relations, dim)
        
        # Initialize embeddings
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize embeddings with Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.ent_re.weight)
        nn.init.xavier_uniform_(self.ent_im.weight)
        nn.init.xavier_uniform_(self.rel_re.weight)
        nn.init.xavier_uniform_(self.rel_im.weight)
    
    def score(self, h_re, h_im, r_re, r_im, t_re, t_im):
        """
        Compute ComplEx score: Re(⟨h, r, conj(t)⟩)
        
        = h_re * r_re * t_re 
        + h_re * r_im * t_im 
        + h_im * r_re * t_im 
        - h_im * r_im * t_re
        
        Returns:
            Scores (higher = more plausible)
        """
        score = (h_re * r_re * t_re +
                 h_re * r_im * t_im +
                 h_im * r_re * t_im -
                 h_im * r_im * t_re)
        return score.sum(dim=-1)
    
    def forward(self, triples: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of triples.
        
        Args:
            triples: Tensor of shape (batch_size, 3) with [head_id, relation_id, tail_id]
            
        Returns:
            Scores for each triple (higher = more plausible)
        """
        h_idx = triples[:, 0]
        r_idx = triples[:, 1]
        t_idx = triples[:, 2]
        
        h_re = self.ent_re(h_idx)
        h_im = self.ent_im(h_idx)
        r_re = self.rel_re(r_idx)
        r_im = self.rel_im(r_idx)
        t_re = self.ent_re(t_idx)
        t_im = self.ent_im(t_idx)
        
        return self.score(h_re, h_im, r_re, r_im, t_re, t_im)
    
    def regularization(self, triples: torch.Tensor) -> torch.Tensor:
        """
        Compute L2 regularization term for embeddings.
        
        Args:
            triples: Tensor of triples used in current batch
            
        Returns:
            Regularization loss
        """
        h_idx = triples[:, 0]
        r_idx = triples[:, 1]
        t_idx = triples[:, 2]
        
        reg = (self.ent_re(h_idx).pow(2).mean() +
               self.ent_im(h_idx).pow(2).mean() +
               self.rel_re(r_idx).pow(2).mean() +
               self.rel_im(r_idx).pow(2).mean() +
               self.ent_re(t_idx).pow(2).mean() +
               self.ent_im(t_idx).pow(2).mean())
        
        return self.reg_weight * reg
    
    def score_all_tails(self, h_idx: torch.Tensor, r_idx: torch.Tensor) -> torch.Tensor:
        """
        Score all possible tail entities for given (head, relation) pairs.
        
        Args:
            h_idx: Head entity indices (batch_size,)
            r_idx: Relation indices (batch_size,)
            
        Returns:
            Scores for all entities as tails (batch_size, num_entities)
        """
        h_re = self.ent_re(h_idx)
        h_im = self.ent_im(h_idx)
        r_re = self.rel_re(r_idx)
        r_im = self.rel_im(r_idx)
        
        all_t_re = self.ent_re.weight
        all_t_im = self.ent_im.weight
        
        scores = (torch.mm(h_re * r_re, all_t_re.t()) +
                  torch.mm(h_re * r_im, all_t_im.t()) +
                  torch.mm(h_im * r_re, all_t_im.t()) -
                  torch.mm(h_im * r_im, all_t_re.t()))
        
        return scores
    
    def score_all_heads(self, r_idx: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        """
        Score all possible head entities for given (relation, tail) pairs.
        
        Args:
            r_idx: Relation indices (batch_size,)
            t_idx: Tail entity indices (batch_size,)
            
        Returns:
            Scores for all entities as heads (batch_size, num_entities)
        """
        r_re = self.rel_re(r_idx)
        r_im = self.rel_im(r_idx)
        t_re = self.ent_re(t_idx)
        t_im = self.ent_im(t_idx)
        
        all_h_re = self.ent_re.weight
        all_h_im = self.ent_im.weight
        
        scores = (torch.mm(r_re * t_re, all_h_re.t()) +
                  torch.mm(r_im * t_im, all_h_re.t()) +
                  torch.mm(r_re * t_im, all_h_im.t()) -
                  torch.mm(r_im * t_re, all_h_im.t()))
        
        return scores


class TriModel(nn.Module):
    """
    TriModel: Tri-vector Embeddings for Knowledge Graph Completion
    
    Uses three embedding vectors per entity and relation, enabling richer
    interactions between components. This model can capture more complex
    patterns than bilinear models like DistMult.
    
    Score function: 
        score(h, r, t) = Σ(h₁ * r₁ * t₃ + h₂ * r₂ * t₂ + h₃ * r₃ * t₁)
    
    Higher scores indicate more plausible triples.
    
    Reference:
        Based on libkge implementation by Sameh Kamaleldin
        https://github.com/samehkamaleldin/libkge
    
    Args:
        num_entities: Number of unique entities in the knowledge graph
        num_relations: Number of unique relation types
        dim: Embedding dimension for each of the 3 components (total = 3*dim per entity/relation)
        reg_weight: L2 regularization weight
    """
    
    def __init__(self, num_entities: int, num_relations: int, dim: int = 100, reg_weight: float = 0.01):
        super().__init__()
        self.dim = dim
        self.reg_weight = reg_weight
        self.num_entities = num_entities
        self.num_relations = num_relations
        
        # Three embedding vectors for entities (v1, v2, v3)
        self.ent_v1 = nn.Embedding(num_entities, dim)
        self.ent_v2 = nn.Embedding(num_entities, dim)
        self.ent_v3 = nn.Embedding(num_entities, dim)
        
        # Three embedding vectors for relations (v1, v2, v3)
        self.rel_v1 = nn.Embedding(num_relations, dim)
        self.rel_v2 = nn.Embedding(num_relations, dim)
        self.rel_v3 = nn.Embedding(num_relations, dim)
        
        # Initialize embeddings
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize embeddings with Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.ent_v1.weight)
        nn.init.xavier_uniform_(self.ent_v2.weight)
        nn.init.xavier_uniform_(self.ent_v3.weight)
        nn.init.xavier_uniform_(self.rel_v1.weight)
        nn.init.xavier_uniform_(self.rel_v2.weight)
        nn.init.xavier_uniform_(self.rel_v3.weight)
    
    def score(self, h_v1, h_v2, h_v3, r_v1, r_v2, r_v3, t_v1, t_v2, t_v3):
        """
        Compute TriModel score: h₁*r₁*t₃ + h₂*r₂*t₂ + h₃*r₃*t₁
        
        Returns:
            Scores (higher = more plausible)
        """
        interaction = (h_v1 * r_v1 * t_v3 +
                       h_v2 * r_v2 * t_v2 +
                       h_v3 * r_v3 * t_v1)
        return interaction.sum(dim=-1)
    
    def forward(self, triples: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of triples.
        
        Args:
            triples: Tensor of shape (batch_size, 3) with [head_id, relation_id, tail_id]
            
        Returns:
            Scores for each triple (higher = more plausible)
        """
        h_idx = triples[:, 0]
        r_idx = triples[:, 1]
        t_idx = triples[:, 2]
        
        h_v1 = self.ent_v1(h_idx)
        h_v2 = self.ent_v2(h_idx)
        h_v3 = self.ent_v3(h_idx)
        
        r_v1 = self.rel_v1(r_idx)
        r_v2 = self.rel_v2(r_idx)
        r_v3 = self.rel_v3(r_idx)
        
        t_v1 = self.ent_v1(t_idx)
        t_v2 = self.ent_v2(t_idx)
        t_v3 = self.ent_v3(t_idx)
        
        return self.score(h_v1, h_v2, h_v3, r_v1, r_v2, r_v3, t_v1, t_v2, t_v3)
    
    def regularization(self, triples: torch.Tensor) -> torch.Tensor:
        """
        Compute L3 (nuclear 3-norm) regularization term for embeddings.
        This is the regularization used in the original TriModel.
        
        Args:
            triples: Tensor of triples used in current batch
            
        Returns:
            Regularization loss
        """
        h_idx = triples[:, 0]
        r_idx = triples[:, 1]
        t_idx = triples[:, 2]
        
        # L3 regularization (nuclear 3-norm as in original implementation)
        reg = (self.ent_v1(h_idx).abs().pow(3).mean() +
               self.ent_v2(h_idx).abs().pow(3).mean() +
               self.ent_v3(h_idx).abs().pow(3).mean() +
               self.rel_v1(r_idx).abs().pow(3).mean() +
               self.rel_v2(r_idx).abs().pow(3).mean() +
               self.rel_v3(r_idx).abs().pow(3).mean() +
               self.ent_v1(t_idx).abs().pow(3).mean() +
               self.ent_v2(t_idx).abs().pow(3).mean() +
               self.ent_v3(t_idx).abs().pow(3).mean())
        
        return (self.reg_weight / 3) * reg
    
    def score_all_tails(self, h_idx: torch.Tensor, r_idx: torch.Tensor) -> torch.Tensor:
        """
        Score all possible tail entities for given (head, relation) pairs.
        
        Args:
            h_idx: Head entity indices (batch_size,)
            r_idx: Relation indices (batch_size,)
            
        Returns:
            Scores for all entities as tails (batch_size, num_entities)
        """
        h_v1 = self.ent_v1(h_idx)
        h_v2 = self.ent_v2(h_idx)
        h_v3 = self.ent_v3(h_idx)
        
        r_v1 = self.rel_v1(r_idx)
        r_v2 = self.rel_v2(r_idx)
        r_v3 = self.rel_v3(r_idx)
        
        all_t_v1 = self.ent_v1.weight
        all_t_v2 = self.ent_v2.weight
        all_t_v3 = self.ent_v3.weight
        
        # score = h₁*r₁*t₃ + h₂*r₂*t₂ + h₃*r₃*t₁
        scores = (torch.mm(h_v1 * r_v1, all_t_v3.t()) +
                  torch.mm(h_v2 * r_v2, all_t_v2.t()) +
                  torch.mm(h_v3 * r_v3, all_t_v1.t()))
        
        return scores
    
    def score_all_heads(self, r_idx: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        """
        Score all possible head entities for given (relation, tail) pairs.
        
        Args:
            r_idx: Relation indices (batch_size,)
            t_idx: Tail entity indices (batch_size,)
            
        Returns:
            Scores for all entities as heads (batch_size, num_entities)
        """
        r_v1 = self.rel_v1(r_idx)
        r_v2 = self.rel_v2(r_idx)
        r_v3 = self.rel_v3(r_idx)
        
        t_v1 = self.ent_v1(t_idx)
        t_v2 = self.ent_v2(t_idx)
        t_v3 = self.ent_v3(t_idx)
        
        all_h_v1 = self.ent_v1.weight
        all_h_v2 = self.ent_v2.weight
        all_h_v3 = self.ent_v3.weight
        
        # score = h₁*r₁*t₃ + h₂*r₂*t₂ + h₃*r₃*t₁
        scores = (torch.mm(r_v1 * t_v3, all_h_v1.t()) +
                  torch.mm(r_v2 * t_v2, all_h_v2.t()) +
                  torch.mm(r_v3 * t_v1, all_h_v3.t()))
        
        return scores
