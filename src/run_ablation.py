"""Run all 4 ablation configs and compile a comparison table.

Configs: full (char-CNN + augmentation), no_char_cnn, no_aug, neither
(= plain baseline). Each config trains from scratch, so this needs real
GPU time for real epoch counts -- pass --epochs low for a quick smoke test
of the loop itself before committing to a full run.

Usage:
  python -m src.run_ablation --config config/default.yaml --out_table ablation_results.md
"""
import argparse
import json
import subprocess
import sys

CONFIGS = [
    ("full", []),
    ("no_char_cnn", ["--no_char_cnn"]),
    ("no_aug", ["--no_aug"]),
    ("baseline (no_char_cnn+no_aug)", ["--no_char_cnn", "--no_aug"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--epochs", type=int, default=None,
                     help="override epochs for all configs (useful for a quick smoke test)")
    ap.add_argument("--ckpt_dir", default="checkpoints/ablation")
    ap.add_argument("--out_table", default="ablation_results.md")
    args = ap.parse_args()

    results = []
    for name, flags in CONFIGS:
        safe_name = name.split()[0]
        out_path = f"{args.ckpt_dir}/{safe_name}.pt"
        cmd = [sys.executable, "-m", "src.train_fused", "--config", args.config,
               "--out", out_path] + flags
        if args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]

        print(f"=== running config: {name}  ({' '.join(flags) or 'no flags'}) ===")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:])
            raise RuntimeError(f"config {name} failed, see output above")

        manifest_path = out_path.replace(".pt", "_manifest.json")
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        final = manifest["history"][-1]
        results.append({
            "config": name,
            "clean_polarity_f1": final["test_clean"]["polarity_f1"],
            "perturbed_polarity_f1": final["test_perturbed"]["polarity_f1"],
            "robustness_delta": final["robustness_delta"],
        })
        print(f"    -> clean={results[-1]['clean_polarity_f1']}  "
              f"perturbed={results[-1]['perturbed_polarity_f1']}  "
              f"delta={results[-1]['robustness_delta']:.2f}")

    with open(args.out_table, "w", encoding="utf-8") as fh:
        fh.write("| Config | Clean polarity_f1 | Perturbed polarity_f1 | Robustness delta |\n")
        fh.write("|---|---|---|---|\n")
        for r in results:
            fh.write(f"| {r['config']} | {r['clean_polarity_f1']} | "
                     f"{r['perturbed_polarity_f1']} | {r['robustness_delta']:.2f} |\n")
    print(f"\nAblation table written to {args.out_table}")


if __name__ == "__main__":
    main()
