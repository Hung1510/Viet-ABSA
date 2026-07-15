"""Shared training infrastructure for train.py and train_fused.py.

Kept in one place so the two scripts can't silently drift apart on things
like seeding or checkpoint-manifest format.
"""
import json
import os
import random

import numpy as np
import torch

from .config import apply_config_defaults


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_common_args(ap):
    """Args shared by train.py and train_fused.py. Call BEFORE any
    script-specific ap.add_argument calls, then use resolve_config after
    the full parser is built."""
    ap.add_argument("--config", default=None,
                     help="YAML file (e.g. config/default.yaml) supplying defaults; "
                          "any CLI flag you also pass overrides the file")
    ap.add_argument("--data_dir", default="data/raw")
    ap.add_argument("--encoder", default="vinai/phobert-base")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_class_weights", action="store_true",
                     help="disable per-aspect class weighting (on by default; "
                          "corrects for rare aspects like STORAGE being drowned "
                          "out in the summed loss)")
    ap.add_argument("--patience", type=int, default=4,
                     help="stop if dev polarity_f1 doesn't improve for this many epochs")
    ap.add_argument("--no_amp", action="store_true",
                     help="disable mixed-precision training (on by default when CUDA "
                          "is available; no effect on CPU)")


def resolve_config(ap):
    """Two-stage parse: read --config first, apply its values as argparse
    defaults, then do the real parse so explicit CLI flags still win."""
    partial, _ = ap.parse_known_args()
    apply_config_defaults(ap, partial)
    return ap.parse_args()


class AMPHelper:
    """Wraps torch.cuda.amp so train.py/train_fused.py don't need an
    if/else at every forward+backward call site. On CPU or with --no_amp,
    autocast/scaler become no-ops."""

    def __init__(self, device: str, enabled: bool):
        self.enabled = enabled and device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.enabled)

    def autocast(self):
        return torch.autocast(device_type="cuda", enabled=self.enabled)

    def backward(self, loss):
        self.scaler.scale(loss).backward()

    def step(self, optimizer):
        self.scaler.step(optimizer)
        self.scaler.update()


def save_manifest(out_path: str, args: dict, history: list):
    """Write <checkpoint>.json next to the .pt file: run args + per-epoch dev
    metrics, so results are reproducible and there's a ready-made source for
    the paper's training curves/tables without re-parsing stdout logs."""
    manifest_path = os.path.splitext(out_path)[0] + "_manifest.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump({"args": args, "history": history}, fh, ensure_ascii=False, indent=2)
    return manifest_path
