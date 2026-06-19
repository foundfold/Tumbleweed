"""Target-conditional binding classifier.

Frozen encoder produces a target-conditional pooled embedding for (sequence, target).
A 3-layer MLP head maps that to a single logit: P(sequence binds target).

Training data:
  - **POSITIVE** : (anchor sequence, anchor's true target_protein) → label 1
  - **NEG-R0**   : (R0 random sequence, any training target_protein) → label 0
  - **NEG-XTGT** : (anchor sequence, DIFFERENT target_protein) → label 0
                  forces the classifier to use target info, not just sequence

For LOO experiments, holdout targets are excluded entirely from the positive
pool — the classifier never sees any of their binders. We then evaluate
zero-shot on the held-out target's test seqs + R0.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class BindingClassifier(nn.Module):
    """3-layer MLP on top of frozen encoder's pooled hidden.

    Args:
        d_in: encoder pooled hidden dim (384 for 14M, 768 for 100M)
        d_hidden: MLP hidden dim
        dropout: dropout between layers
    """
    def __init__(self, d_in: int = 384, d_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """pooled: [B, d_in] → logits [B] (pre-sigmoid)."""
        return self.net(pooled).squeeze(-1)

    def predict_proba(self, pooled: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(pooled))


def bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, labels.float())


if __name__ == '__main__':
    clf = BindingClassifier()
    n = sum(p.numel() for p in clf.parameters())
    print(f'BindingClassifier: {n/1e6:.2f}M params')
    fake = torch.randn(8, 384)
    logits = clf(fake)
    probs = clf.predict_proba(fake)
    print(f'logits shape: {logits.shape}, probs: {probs.tolist()}')
