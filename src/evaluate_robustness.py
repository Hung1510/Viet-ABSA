"""Robustness evaluation: severity sweep, per-aspect delta, and bootstrap
significance testing -- compares a checkpoint's clean vs. perturbed
performance, and optionally two checkpoints against each other.

Usage (single model, single severity level -- same as before):
  python -m src.evaluate_robustness --model baseline --checkpoint checkpoints/baseline.pt

Severity sweep + curve:
  python -m src.evaluate_robustness --model fused --checkpoint checkpoints/fused.pt \
      --perturb_level 0.1 0.3 0.5 0.7 0.9 --curve_out figures/robustness_curve.png

Statistical comparison of two models (the actual "does char-CNN help" test):
  python -m src.evaluate_robustness --model fused --checkpoint checkpoints/fused.pt \
      --compare_model baseline --compare_checkpoint checkpoints/baseline.pt \
      --bootstrap --n_bootstrap 1000
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .data.dataset import ViSFDDataset
from .data.perturb import augment_fixed
from .models.baseline import ViSFDBaseline
from .models.fused import FusedViSFDModel
from .evaluate import collect_preds, compute_metrics
from .train_fused import collect_preds_fused


def load_model_and_preds(model_name, checkpoint, encoder, data_dir, max_len, batch,
                          device, perturb_level=None):
    """Returns (preds, labels) for the clean test set, and for a perturbed
    copy at `perturb_level` if given (else perturbed=None)."""
    tok = AutoTokenizer.from_pretrained(encoder, use_fast=False)
    include_chars = model_name == "fused"

    clean_ds = ViSFDDataset(f"{data_dir}/Test.csv", tok, max_len=max_len, include_chars=include_chars)
    clean_loader = DataLoader(clean_ds, batch_size=batch, shuffle=False, num_workers=2)

    if model_name == "baseline":
        model = ViSFDBaseline(encoder).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        p_clean, y = collect_preds(model, clean_loader, device)
    else:
        model = FusedViSFDModel(encoder).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        p_clean, y = collect_preds_fused(model, clean_loader, device)

    p_pert = None
    if perturb_level is not None:
        pert_ds = ViSFDDataset(f"{data_dir}/Test.csv", tok, max_len=max_len,
                                include_chars=include_chars,
                                augment_fn=lambda t: augment_fixed(t, level=perturb_level))
        pert_loader = DataLoader(pert_ds, batch_size=batch, shuffle=False, num_workers=2)
        if model_name == "baseline":
            p_pert, _ = collect_preds(model, pert_loader, device)
        else:
            p_pert, _ = collect_preds_fused(model, pert_loader, device)

    return p_clean, p_pert, y


def bootstrap_delta_pvalue(preds_clean, preds_pert, labels, n_bootstrap=1000, seed=0):
    """One-sample bootstrap: is (clean_f1 - perturbed_f1) significantly > 0?
    Returns (mean_delta, ci_lo, ci_hi, p_value) where p_value is the fraction
    of resamples where delta <= 0 (i.e. evidence AGAINST perturbation hurting)."""
    rng = np.random.default_rng(seed)
    n = labels.shape[0]
    deltas = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        idx_t = torch.as_tensor(idx)
        m_clean = compute_metrics(preds_clean[idx_t], labels[idx_t])
        m_pert = compute_metrics(preds_pert[idx_t], labels[idx_t])
        deltas[b] = m_clean["polarity_f1"] - m_pert["polarity_f1"]
    mean_delta = float(deltas.mean())
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    p_value = float((deltas <= 0).mean())
    return mean_delta, float(ci_lo), float(ci_hi), p_value


def bootstrap_compare_pvalue(preds_clean_a, preds_pert_a, preds_clean_b, preds_pert_b,
                              labels, n_bootstrap=1000, seed=0):
    """Paired bootstrap comparing two models' robustness deltas (same resampled
    indices used for both, so the comparison is paired, not independent).
    model B is the one expected to be MORE robust (smaller delta), e.g. fused.
    Returns (mean_diff = delta_b - delta_a, ci_lo, ci_hi, p_value) where
    p_value is the fraction of resamples where delta_b >= delta_a (i.e.
    evidence AGAINST "B is more robust than A")."""
    rng = np.random.default_rng(seed)
    n = labels.shape[0]
    diffs = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        idx_t = torch.as_tensor(idx)
        y = labels[idx_t]
        delta_a = (compute_metrics(preds_clean_a[idx_t], y)["polarity_f1"]
                   - compute_metrics(preds_pert_a[idx_t], y)["polarity_f1"])
        delta_b = (compute_metrics(preds_clean_b[idx_t], y)["polarity_f1"]
                   - compute_metrics(preds_pert_b[idx_t], y)["polarity_f1"])
        diffs[b] = delta_b - delta_a
    mean_diff = float(diffs.mean())
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    p_value = float((diffs >= 0).mean())
    return mean_diff, float(ci_lo), float(ci_hi), p_value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "fused"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--compare_model", choices=["baseline", "fused"], default=None,
                     help="optional second model to statistically compare against")
    ap.add_argument("--compare_checkpoint", default=None)
    ap.add_argument("--data_dir", default="data/raw")
    ap.add_argument("--encoder", default="vinai/phobert-base")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--perturb_level", type=float, nargs="+", default=[0.5],
                     help="one or more severity levels in [0,1]; >1 value triggers a sweep")
    ap.add_argument("--curve_out", default=None,
                     help="path to save robustness_curve.png (only used if >1 perturb_level given)")
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--n_bootstrap", type=int, default=1000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sweep_results = {"levels": args.perturb_level, "polarity_f1": []}
    p_clean = p_pert = labels = None
    for level in args.perturb_level:
        p_clean, p_pert, labels = load_model_and_preds(
            args.model, args.checkpoint, args.encoder, args.data_dir,
            args.max_len, args.batch, device, perturb_level=level)
        m_clean = compute_metrics(p_clean, labels)
        m_pert = compute_metrics(p_pert, labels)
        delta = m_clean["polarity_f1"] - m_pert["polarity_f1"]
        sweep_results["polarity_f1"].append(m_pert["polarity_f1"])
        print(f"[{args.model}] level={level:.2f}  clean={m_clean['polarity_f1']}  "
              f"perturbed={m_pert['polarity_f1']}  delta={delta:.2f}")
        print(f"           per-aspect (clean):     {m_clean['per_aspect_f1']}")
        print(f"           per-aspect (perturbed): {m_pert['per_aspect_f1']}")

    if len(args.perturb_level) > 1 and args.curve_out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot(sweep_results["levels"], sweep_results["polarity_f1"], marker="o", label=args.model)
        plt.xlabel("Perturbation severity")
        plt.ylabel("Perturbed test polarity_f1")
        plt.title("Robustness curve: F1 vs. noise intensity")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.curve_out, dpi=150)
        plt.close()
        print(f"Saved robustness curve to {args.curve_out}")

    if args.bootstrap:
        # bootstrap always uses the LAST level evaluated above (or the single
        # level if only one was given) -- for a clean one-number significance
        # claim, pass a single --perturb_level alongside --bootstrap.
        mean_delta, lo, hi, p = bootstrap_delta_pvalue(p_clean, p_pert, labels, args.n_bootstrap)
        print(f"[{args.model}] bootstrap delta: mean={mean_delta:.2f} "
              f"95% CI=({lo:.2f}, {hi:.2f})  p(delta<=0)={p:.4f}")

        if args.compare_model and args.compare_checkpoint:
            cp_clean, cp_pert, cp_labels = load_model_and_preds(
                args.compare_model, args.compare_checkpoint, args.encoder, args.data_dir,
                args.max_len, args.batch, device, perturb_level=args.perturb_level[-1])
            # sanity: label order must match (same Test.csv, same row order)
            assert torch.equal(labels, cp_labels), "label mismatch between models -- did data_dir change?"
            mean_diff, lo2, hi2, p2 = bootstrap_compare_pvalue(
                cp_clean, cp_pert, p_clean, p_pert, labels, args.n_bootstrap)
            print(f"[{args.model} vs {args.compare_model}] delta_diff mean={mean_diff:.2f} "
                  f"95% CI=({lo2:.2f}, {hi2:.2f})  p(no improvement)={p2:.4f}")
            if p2 < 0.05:
                print(f"  -> {args.model}'s robustness delta is significantly smaller than "
                      f"{args.compare_model}'s (p < 0.05)")
            else:
                print(f"  -> not statistically significant at p < 0.05 -- don't claim significance")


if __name__ == "__main__":
    main()
