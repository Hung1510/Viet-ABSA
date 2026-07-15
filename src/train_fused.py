"""Train the fused PhoBERT+char-CNN model with adversarial augmentation.

DO NOT run this before train.py's baseline has reproduced the target numbers
(see viet-absa-skeleton-and-handoff.md build order). This script depends on
baseline.py's PhoBERT branch being validated first; if results here look
wrong, first confirm the plain baseline still reproduces before debugging
anything in this file.

Reports the robustness delta (clean test macro-F1 vs. perturbed test
macro-F1) for THIS model. Run evaluate_robustness.py afterwards on both the
baseline and fused checkpoints to compare deltas directly -- that comparison,
not the raw score, is the actual contribution claim.

Usage:
  python -m src.train_fused --config config/default.yaml
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .data.dataset import ViSFDDataset, compute_class_weights
from .data.perturb import augment, augment_fixed
from .models.fused import FusedViSFDModel
from .evaluate import compute_metrics
from .train_utils import set_seed, add_common_args, resolve_config, AMPHelper, save_manifest


def build_loader(csv_path, tok, max_len, batch, shuffle, augment_fn=None):
    ds = ViSFDDataset(csv_path, tok, max_len=max_len, include_chars=True,
                       augment_fn=augment_fn)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=2)


@torch.no_grad()
def collect_preds_fused(model, loader, device):
    model.eval()
    P, Y = [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        chars = batch["char_ids"].to(device)
        logits = model(ids, mask, chars)
        preds = torch.stack([l.argmax(-1) for l in logits], dim=1)
        P.append(preds.cpu())
        Y.append(batch["labels"])
    return torch.cat(P), torch.cat(Y)


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--out", default="checkpoints/fused.pt")
    ap.add_argument("--augment_p", type=float, default=0.5,
                     help="probability of applying augmentation per train sample")
    args = resolve_config(ap)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = AMPHelper(device, enabled=not args.no_amp)
    tok = AutoTokenizer.from_pretrained(args.encoder, use_fast=False)

    train_augment = lambda t: augment(t, p=args.augment_p)
    tr = build_loader(f"{args.data_dir}/Train.csv", tok, args.max_len, args.batch,
                       True, augment_fn=train_augment)
    dev = build_loader(f"{args.data_dir}/Dev.csv", tok, args.max_len, args.batch, False)
    te_clean = build_loader(f"{args.data_dir}/Test.csv", tok, args.max_len, args.batch, False)
    te_perturbed = build_loader(f"{args.data_dir}/Test.csv", tok, args.max_len, args.batch,
                                 False, augment_fn=augment_fixed)

    class_weights = None
    if not args.no_class_weights:
        class_weights = compute_class_weights(f"{args.data_dir}/Train.csv").to(device)

    model = FusedViSFDModel(args.encoder).to(device)
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
            chars = batch["char_ids"].to(device)
            labels = batch["labels"].to(device)
            opt.zero_grad()
            with amp.autocast():
                logits = model(ids, mask, chars)
                loss = model.loss(logits, labels, class_weights=class_weights)
            amp.backward(loss)
            amp.scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp.step(opt)
            sched.step()
            run += loss.item()

        p, y = collect_preds_fused(model, dev, device)
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
            save_manifest(args.out, vars(args), history)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"no dev improvement for {args.patience} epochs, stopping early at epoch {ep}")
                break

    model.load_state_dict(torch.load(args.out))
    p_clean, y_clean = collect_preds_fused(model, te_clean, device)
    p_pert, y_pert = collect_preds_fused(model, te_perturbed, device)
    m_clean = compute_metrics(p_clean, y_clean)
    m_pert = compute_metrics(p_pert, y_pert)
    delta = m_clean["polarity_f1"] - m_pert["polarity_f1"]

    print("TEST (clean):    ", m_clean)
    print("TEST (perturbed):", m_pert)
    print(f"ROBUSTNESS DELTA (polarity_f1, clean - perturbed): {delta:.2f} "
          "-- lower is better (more robust). Run evaluate_robustness.py on "
          "the plain baseline checkpoint too, to compare this delta directly.")

    manifest_path = save_manifest(
        args.out, vars(args),
        history + [{"test_clean": m_clean, "test_perturbed": m_pert, "robustness_delta": delta}],
    )
    print(f"manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
