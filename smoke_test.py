"""Smoke test: verify label parsing + Dataset against REAL UIT-ViSFD rows.
Uses a stub tokenizer so no HuggingFace download is needed here."""
import csv, torch
from src.data.labels import parse_label, ASPECTS, ID_TO_POLARITY, NUM_ASPECTS
from src.data.dataset import ViSFDDataset

DATA = "data/raw"

# --- 1. parse_label on known rows -------------------------------------------
cases = {
    "{CAMERA#Positive};{FEATURES#Positive};{BATTERY#Positive};{PRICE#Positive};{GENERAL#Positive};{SER&ACC#Positive};":
        [1,1,0,1,1,0,1,0,1,0],
    "{BATTERY#Negative};{GENERAL#Positive};{OTHERS};":
        [3,0,0,0,1,0,0,0,0,0],
    "{DESIGN#Neutral};{SER&ACC#Positive};":
        [0,0,2,0,0,0,0,0,1,0],
}
for lab, exp in cases.items():
    got = parse_label(lab)
    assert got == exp, f"MISMATCH\n {lab}\n exp {exp}\n got {got}"
print("[1] parse_label: 3/3 exact matches on real label strings  OK")

# --- 2. round-trip check across the whole train split -----------------------
bad = 0
with open(f"{DATA}/Train.csv", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        st = parse_label(row["label"])
        # every non-zero state must correspond to an aspect present in the raw label
        for idx, s in enumerate(st):
            if s and ASPECTS[idx] not in row["label"]:
                bad += 1
print(f"[2] full-train consistency: {bad} contradictions  {'OK' if bad==0 else 'FAIL'}")

# --- 3. Dataset shapes with a stub tokenizer --------------------------------
class StubTok:
    def __call__(self, text, truncation, max_length, padding, return_tensors):
        ids = [1] + [hash(w) % 60000 for w in text.split()][:max_length-1]
        ids = ids[:max_length] + [0]*(max_length-len(ids))
        mask = [1 if t else 0 for t in ids]
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([mask])}

ds = ViSFDDataset(f"{DATA}/Train.csv", StubTok(), max_len=64)
item = ds[0]
print(f"[3] dataset len={len(ds)}  input_ids={tuple(item['input_ids'].shape)} "
      f"labels={tuple(item['labels'].shape)} dtype={item['labels'].dtype}")
assert item["input_ids"].shape == (64,)
assert item["labels"].shape == (NUM_ASPECTS,)
assert item["labels"].tolist() == ds.samples[0][1]
print("    dataset shapes  OK")

# --- 4. label distribution sanity -------------------------------------------
from collections import Counter
c = Counter()
for _, st in ds.samples:
    for idx, s in enumerate(st):
        if s: c[ASPECTS[idx]] += 1
print("[4] aspect frequency (train):",
      {k: c[k] for k in sorted(c, key=c.get, reverse=True)})
