"""Character-level CNN branch -- the contribution on top of the PhoBERT baseline.

Deliberately fed the RAW surface form (post-normalize, pre-word-segmentation),
not the PhoBERT branch's segmented text. Rationale: typos, dropped tone marks,
and teencode substitutions (ko/dc/wa) live at the character/byte level, and
VnCoreNLP/underthesea word segmentation would already (mis)tokenize around
them before this branch ever saw them -- segmenting first would throw away
exactly the signal this branch exists to capture.

Uses raw UTF-8 bytes (vocab=256) rather than a learned character vocabulary:
- never produces an OOV token, including for emoji and mixed Latin/Vietnamese
- Vietnamese diacritics are multi-byte UTF-8 sequences, so byte-level still
  distinguishes "a" from "a" + combining marks / precomposed diacritic forms
- no vocab-building step needed before training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def text_to_byte_ids(text: str, max_chars: int = 300) -> list:
    """Encode `text` as UTF-8 bytes, 0 reserved for padding (bytes shifted +1)."""
    raw = text.encode("utf-8")[:max_chars]
    ids = [b + 1 for b in raw]
    ids += [0] * (max_chars - len(ids))
    return ids


class CharCNN(nn.Module):
    """Multi-kernel 1D conv over byte embeddings, max-pooled per kernel width.

    Standard Kim (2014)-style char/word-CNN: each kernel width captures a
    different obfuscation pattern (k=2 catches single adjacent-key typos,
    k=3-4 catch short teencode substrings like 'ko', 'dc', 'qua').
    """

    def __init__(self, vocab_size=256, emb_dim=64, channels=128, kernels=(2, 3, 4)):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim, channels, kernel_size=k, padding=k // 2)
            for k in kernels
        ])
        self.out_dim = channels * len(kernels)

    def forward(self, char_ids):                        # [B, T_chars]
        x = self.embed(char_ids).transpose(1, 2)          # [B, E, T]
        pooled = [
            F.adaptive_max_pool1d(F.relu(conv(x)), 1).squeeze(-1)  # [B, C]
            for conv in self.convs
        ]
        return torch.cat(pooled, dim=-1)                  # [B, C*len(kernels)]
