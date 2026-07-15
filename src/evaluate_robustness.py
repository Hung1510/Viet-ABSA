"""Compute robustness delta (clean vs. perturbed test macro-F1) for a
trained checkpoint -- baseline or fused -- so the two can be compared
side by side. This comparison, not either model's raw score, is what
supports the char-CNN contribution claim.

Usage:
  python -m src.evaluate_robustness --model baseline --checkpoint checkpoints/baseline.pt
  python -m src.evaluate_robustness --model fused    --checkpoint checkpoints/fused.pt
"""
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .data.dataset import ViSFDDataset
from .data.perturb import augment_fixed
from .models.baseline import ViSFDBaseline
from .models.fused import FusedViSFDModel
from .evaluate import collect_preds, compute_metrics
from .train_fused import collect_preds_fused


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "fused"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", default="data/raw")
    ap.add_argument("--encoder", default="vinai/phobert-base")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.encoder, use_fast=False)
    include_chars = args.model == "fused"

    clean_ds = ViSFDDataset(f"{args.data_dir}/Test.csv", tok, max_len=args.max_len,
                             include_chars=include_chars)
    pert_ds = ViSFDDataset(f"{args.data_dir}/Test.csv", tok, max_len=args.max_len,
                            include_chars=include_chars, augment_fn=augment_fixed)
    clean_loader = DataLoader(clean_ds, batch_size=args.batch, shuffle=False, num_workers=2)
    pert_loader = DataLoader(pert_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    if args.model == "baseline":
        model = ViSFDBaseline(args.encoder).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        p_clean, y_clean = collect_preds(model, clean_loader, device)
        p_pert, y_pert = collect_preds(model, pert_loader, device)
    else:
        model = FusedViSFDModel(args.encoder).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        p_clean, y_clean = collect_preds_fused(model, clean_loader, device)
        p_pert, y_pert = collect_preds_fused(model, pert_loader, device)

    m_clean = compute_metrics(p_clean, y_clean)
    m_pert = compute_metrics(p_pert, y_pert)
    delta = m_clean["polarity_f1"] - m_pert["polarity_f1"]

    print(f"[{args.model}] TEST (clean):    ", m_clean)
    print(f"[{args.model}] TEST (perturbed):", m_pert)
    print(f"[{args.model}] ROBUSTNESS DELTA (clean - perturbed): {delta:.2f} "
          "(lower = more robust)")


if __name__ == "__main__":
    main()
