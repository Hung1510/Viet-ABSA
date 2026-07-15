"""Single-comment inference -- useful for qualitative spot-checks during
development, and as the reference implementation for the future FastAPI
endpoint (api/main.py, not yet built).

Usage:
  python -m src.infer --model baseline --checkpoint checkpoints/baseline.pt \
      --text "Pin trâu, camera đẹp nhưng giá hơi cao"
"""
import argparse
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
    ap.add_argument("--text", required=True)
    ap.add_argument("--max_len", type=int, default=256)
    args = ap.parse_args()

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

    result = predict(args.text, model, tok, device, is_fused, args.max_len)
    print(f"Text: {args.text}")
    if not result:
        print("No aspects detected.")
    for aspect, polarity in result.items():
        print(f"  {aspect:12s} -> {polarity}")


if __name__ == "__main__":
    main()
