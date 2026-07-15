"""Single-comment or batch inference -- useful for qualitative spot-checks
during development, and as the reference implementation for the future
FastAPI endpoint (api/main.py, not yet built).

Usage (single comment):
  python -m src.infer --model baseline --checkpoint checkpoints/baseline.pt \
      --text "Pin trâu, camera đẹp nhưng giá hơi cao"

Usage (batch, e.g. a CSV of unlabeled comments with a "comment" column):
  python -m src.infer --model fused --checkpoint checkpoints/fused.pt \
      --input_file new_comments.csv --output_file predictions.csv
"""
import argparse
import csv
import torch

from .data.labels import ASPECTS, ID_TO_POLARITY
from .data.preprocess import preprocess, normalize
from .models.baseline import ViSFDBaseline
from .models.fused import FusedViSFDModel
from .models.char_cnn import text_to_byte_ids


def predict(text, model, tokenizer, device, is_fused, max_len=256, max_chars=300):
    enc = tokenizer(preprocess(text), truncation=True, max_length=max_len,
                     padding="max_length", return_tensors="pt")
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        if is_fused:
            char_ids = torch.tensor([text_to_byte_ids(normalize(text), max_chars)]).to(device)
            logits = model(ids, mask, char_ids)
        else:
            logits = model(ids, mask)
        preds = [l.argmax(-1).item() for l in logits]
    return {ASPECTS[a]: ID_TO_POLARITY[p] for a, p in enumerate(preds) if p != 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "fused"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--encoder", default="vinai/phobert-base")
    ap.add_argument("--text", default=None, help="single-comment mode")
    ap.add_argument("--input_file", default=None,
                     help="batch mode: CSV with a 'comment' column")
    ap.add_argument("--output_file", default=None,
                     help="batch mode: where to write predictions CSV")
    ap.add_argument("--max_len", type=int, default=256)
    args = ap.parse_args()

    if args.text is None and args.input_file is None:
        raise ValueError("pass either --text (single comment) or --input_file (batch CSV)")
    if args.input_file and not args.output_file:
        raise ValueError("--input_file requires --output_file")

    from transformers import AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.encoder, use_fast=False)

    is_fused = args.model == "fused"
    if is_fused:
        model = FusedViSFDModel(args.encoder).to(device)
    else:
        model = ViSFDBaseline(args.encoder).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    if args.text is not None:
        result = predict(args.text, model, tok, device, is_fused, args.max_len)
        print(f"Text: {args.text}")
        if not result:
            print("No aspects detected.")
        for aspect, polarity in result.items():
            print(f"  {aspect:12s} -> {polarity}")
        return

    # batch mode
    with open(args.input_file, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or "comment" not in rows[0]:
        raise ValueError("--input_file must be a CSV with a 'comment' column")

    out_rows = []
    for row in rows:
        result = predict(row["comment"], model, tok, device, is_fused, args.max_len)
        out_row = {"comment": row["comment"]}
        for aspect in ASPECTS:
            out_row[aspect] = result.get(aspect, "none")
        out_rows.append(out_row)

    with open(args.output_file, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["comment"] + ASPECTS)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} predictions to {args.output_file}")


if __name__ == "__main__":
    main()
