"""PhoBERT + char-CNN fusion for multi-task ABSA (the contribution model).

Only meant to be used AFTER baseline.py's PhoBERT-only model reproduces the
~86%/~78.8% target (see viet-absa-skeleton-and-handoff.md build order,
step 4). Mirrors ViSFDBaseline's interface on purpose (forward returns
list[NUM_ASPECTS] of [B,4] logits, same .loss signature) so a training loop
can switch between the two with minimal changes -- see train_fused.py.
"""
import torch
import torch.nn as nn
from transformers import AutoModel

from .baseline import masked_mean, MultiTaskHead, ViSFDBaseline
from .char_cnn import CharCNN


class FusedViSFDModel(nn.Module):
    def __init__(self, encoder_name="vinai/phobert-base", dropout=0.1,
                 char_vocab=256, char_emb=64, char_channels=128, char_kernels=(2, 3, 4)):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.char_cnn = CharCNN(char_vocab, char_emb, char_channels, char_kernels)
        fused_dim = hidden + self.char_cnn.out_dim
        self.head = MultiTaskHead(fused_dim, p=dropout)

    def forward(self, input_ids, attention_mask, char_ids):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        bert_feat = masked_mean(out.last_hidden_state, attention_mask)   # [B, hidden]
        char_feat = self.char_cnn(char_ids)                              # [B, char_out]
        fused = torch.cat([bert_feat, char_feat], dim=-1)                # [B, hidden+char_out]
        return self.head(fused)

    @staticmethod
    def loss(logits, labels, class_weights=None):
        return ViSFDBaseline.loss(logits, labels, class_weights)
