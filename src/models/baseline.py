"""PhoBERT baseline for multi-task ABSA (reproduce BEFORE adding char-CNN).

Encoder (PhoBERT) -> masked mean-pool -> 10 independent 4-way heads.
This is the baseline that targets ~86% aspect / ~78.8% sentiment macro-F1.
Add the char-CNN branch (your contribution) only after this reproduces.
"""
import torch
import torch.nn as nn
from transformers import AutoModel

from ..data.labels import NUM_ASPECTS, NUM_STATES


def masked_mean(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    return summed / counts


class MultiTaskHead(nn.Module):
    """One 4-way classifier per aspect -> list of [B, NUM_STATES]."""
    def __init__(self, in_dim, n_aspects=NUM_ASPECTS, n_states=NUM_STATES, p=0.1):
        super().__init__()
        self.drop = nn.Dropout(p)
        self.heads = nn.ModuleList(nn.Linear(in_dim, n_states) for _ in range(n_aspects))

    def forward(self, feats):
        feats = self.drop(feats)
        return [h(feats) for h in self.heads]


class ViSFDBaseline(nn.Module):
    def __init__(self, encoder_name="vinai/phobert-base", dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.head = MultiTaskHead(hidden, p=dropout)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = masked_mean(out.last_hidden_state, attention_mask)
        return self.head(pooled)                      # list[NUM_ASPECTS] of [B, 4]

    @staticmethod
    def loss(logits, labels, class_weights=None):       # labels: [B, NUM_ASPECTS]
        """class_weights: optional [NUM_ASPECTS, NUM_STATES] tensor from
        dataset.compute_class_weights(), moved to the same device as logits.
        Without it, aspects with few examples (e.g. STORAGE) contribute
        almost nothing to total loss and the model learns to mostly ignore
        them; per-aspect weighting corrects for that."""
        ce = nn.functional.cross_entropy
        if class_weights is None:
            return sum(ce(logits[i], labels[:, i]) for i in range(len(logits)))
        return sum(
            ce(logits[i], labels[:, i], weight=class_weights[i])
            for i in range(len(logits))
        )
