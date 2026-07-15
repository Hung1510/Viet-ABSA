"""Find the worst-predicted aspects and dump misclassified test examples to
CSV for manual inspection.

Usage:
  python -m src.analyze_errors --model baseline --checkpoint checkpoints/baseline.pt \
      --f1_threshold 50 --out errors.csv
"""
import argparse
import csv

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .data.dataset import ViSFDDataset
from .data.labels import ASPECTS, ID_TO_POLARITY
from .models.baseline import ViSFDBaseline
from .models.fused import FusedViSFDModel
from .evaluate import compute_metrics
from .train_fused import collect_preds_fused
from .evaluate import collect_preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "fused"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", default="data/raw")
    ap.add_argument("--encoder", default="vinai/phobert-base")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--f1_threshold", type=float, default=50.0,
                     help="flag aspects with per_aspect_f1 below this (0-100 scale)")
    ap.add_argument("--out", default="errors.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.encoder, use_fast=False)
    include_chars = args.model == "fused"

    ds = ViSFDDataset(f"{args.data_dir}/Test.csv", tok, max_len=args.max_len,
                       include_chars=include_chars)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2)

    if args.model == "baseline":
        model = ViSFDBaseline(args.encoder).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        preds, labels = collect_preds(model, loader, device)
    else:
        model = FusedViSFDModel(args.encoder).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        preds, labels = collect_preds_fused(model, loader, device)

    metrics = compute_metrics(preds, labels)
    weak_aspects = [a for a, f1 in metrics["per_aspect_f1"].items() if f1 < args.f1_threshold]
    print(f"Aspects below F1={args.f1_threshold}: {weak_aspects or 'none'}")

    # ds.samples is (text, states) in the same order the DataLoader iterated
    # (shuffle=False), so row i of preds/labels lines up with ds.samples[i]
    rows = []
    for i, (text, true_states) in enumerate(ds.samples):
        for a_idx, aspect in enumerate(ASPECTS):
            if aspect not in weak_aspects:
                continue
            true_s = labels[i, a_idx].item()
            pred_s = preds[i, a_idx].item()
            if true_s != pred_s:
                rows.append({
                    "aspect": aspect,
                    "text": text,
                    "true": ID_TO_POLARITY[true_s],
                    "predicted": ID_TO_POLARITY[pred_s],
                })

    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["aspect", "text", "true", "predicted"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} misclassified examples to {args.out}")


if __name__ == "__main__":
    main()
