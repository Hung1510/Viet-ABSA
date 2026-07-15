"""Plot training curves (loss, aspect_f1, polarity_f1) from a manifest.json.

Usage:
  python -m src.plot_curves --manifest checkpoints/baseline_manifest.json --out_dir figures/baseline
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_history(manifest_path):
    with open(manifest_path, encoding="utf-8") as fh:
        data = json.load(fh)
    # history entries are per-epoch dicts; trailing entries (test/robustness
    # summaries) don't have an "epoch" key -- filter those out for the curves
    return [h for h in data["history"] if "epoch" in h]


def plot_metric(history, key, ylabel, out_path, dev_label="dev"):
    epochs = [h["epoch"] for h in history]
    values = [h[key] for h in history]
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, values, marker="o", label=dev_label)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} over training")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", default="figures")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    history = load_history(args.manifest)
    if not history:
        print("No per-epoch history found in manifest -- nothing to plot.")
        return

    plot_metric(history, "train_loss", "Train loss", os.path.join(args.out_dir, "loss.png"))
    plot_metric(history, "aspect_f1", "Dev aspect F1", os.path.join(args.out_dir, "aspect_f1.png"))
    plot_metric(history, "polarity_f1", "Dev polarity F1", os.path.join(args.out_dir, "polarity_f1.png"))
    print(f"Saved loss.png, aspect_f1.png, polarity_f1.png to {args.out_dir}")


if __name__ == "__main__":
    main()
