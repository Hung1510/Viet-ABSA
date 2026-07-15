"""Reproduce the PhoBERT baseline on UIT-ViSFD.

Usage:
  python -m src.train --config config/default.yaml
  # or override anything from the CLI:
  python -m src.train --config config/default.yaml --epochs 20 --batch 16

Targets ~86% aspect / ~78.8% polarity macro-F1 (see Le et al. 2022, KSE'22 --
NOT the original SA2SL paper, which reports 84.48%/63.06%. Treat the 86/78.8
figures as a directional target until you've verified their exact macro-F1
definition against your evaluate.py). Run on GPU.
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .data.dataset import ViSFDDataset, compute_class_weights
from .models.baseline import ViSFDBaseline
from .evaluate import collect_preds, compute_metrics
from .train_utils import set_seed, add_common_args, resolve_config, AMPHelper, save_manifest


def build_loader(csv_path, tok, max_len, batch, shuffle):
    ds = ViSFDDataset(csv_path, tok, max_len=max_len)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=2)


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--out", default="checkpoints/baseline.pt")
    args = resolve_config(ap)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = AMPHelper(device, enabled=not args.no_amp)
    tok = AutoTokenizer.from_pretrained(args.encoder, use_fast=False)

    tr = build_loader(f"{args.data_dir}/Train.csv", tok, args.max_len, args.batch, True)
    dev = build_loader(f"{args.data_dir}/Dev.csv", tok, args.max_len, args.batch, False)
    te = build_loader(f"{args.data_dir}/Test.csv", tok, args.max_len, args.batch, False)

    class_weights = None
    if not args.no_class_weights:
        class_weights = compute_class_weights(f"{args.data_dir}/Train.csv").to(device)

    model = ViSFDBaseline(args.encoder).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(tr) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(total * args.warmup), total)

    best = -1.0
    stale = 0
    history = []
    for ep in range(1, args.epochs + 1):
        model.train()
        run = 0.0
        for batch in tr:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            opt.zero_grad()
            with amp.autocast():
                logits = model(ids, mask)
                loss = model.loss(logits, labels, class_weights=class_weights)
            amp.backward(loss)
            amp.scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp.step(opt)
            sched.step()
            run += loss.item()

        p, y = collect_preds(model, dev, device)
        m = compute_metrics(p, y)
        print(f"epoch {ep:2d} | loss {run/len(tr):.3f} | dev aspect_f1={m['aspect_f1']} "
              f"polarity_f1={m['polarity_f1']}")
        print(f"           per-aspect F1: {m['per_aspect_f1']}")
        history.append({"epoch": ep, "train_loss": run / len(tr), **m})

        if m["polarity_f1"] > best:
            best = m["polarity_f1"]
            stale = 0
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save(model.state_dict(), args.out)
            save_manifest(args.out, vars(args), history)  # keep manifest current mid-run
        else:
            stale += 1
            if stale >= args.patience:
                print(f"no dev improvement for {args.patience} epochs, stopping early at epoch {ep}")
                break

    model.load_state_dict(torch.load(args.out))
    p, y = collect_preds(model, te, device)
    test_metrics = compute_metrics(p, y)
    print("TEST:", test_metrics)
    manifest_path = save_manifest(args.out, vars(args), history + [{"test": test_metrics}])
    print(f"manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
