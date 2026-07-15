# viet-absa

Multi-task Aspect-Based Sentiment Analysis for Vietnamese e-commerce reviews.
For each comment, predicts polarity (`none` / `positive` / `neutral` / `negative`)
across 10 smartphone-review aspect categories (UIT-ViSFD schema).

Two models live here:
- **Baseline** (`src/train.py`) — PhoBERT encoder + 10 independent 4-way heads.
  Targets ~86.03% aspect / ~78.76% polarity macro-F1, per Le et al. (2022,
  KSE'22 IEEE). **Note:** that exact metric definition hasn't been verified
  against `evaluate.py` (paper is paywalled) — treat it as a directional
  target, not a strict pass/fail bar, unless you check the original PDF.
- **Fused** (`src/train_fused.py`) — adds a character-level CNN branch (raw
  UTF-8 bytes, robust to typos/teencode/obfuscation) plus adversarial
  augmentation on the train split. This is the paper's actual contribution:
  the **robustness delta** between clean and perturbed test performance.

**Build order — do not skip ahead:** baseline must reproduce its target
numbers *before* you touch the fused model. If the fused model's numbers
look wrong, first confirm the plain baseline still works before debugging
anything in `train_fused.py`.

---

## Repo layout

```
viet-absa/
├── config/default.yaml       # all hyperparams in one place
├── data/raw/{Train,Dev,Test}.csv   # UIT-ViSFD, verified clean (no dupes, no leakage)
├── requirements.txt
├── smoke_test.py              # local sanity check, no GPU/download needed
├── tests/test_perturb.py      # unit tests for the augmentation functions
├── src/
│   ├── config.py               # YAML -> argparse defaults
│   ├── train_utils.py          # shared seed/AMP/manifest logic
│   ├── data/
│   │   ├── labels.py            # aspect/polarity schema + label parsing
│   │   ├── preprocess.py        # normalization + PhoBERT word segmentation
│   │   ├── dataset.py           # torch Dataset (+ char ids, + augmentation)
│   │   └── perturb.py           # typo / tone-drop / teencode augmentation
│   ├── models/
│   │   ├── baseline.py          # PhoBERT-only model
│   │   ├── char_cnn.py          # char-CNN branch
│   │   └── fused.py             # PhoBERT + char-CNN fusion
│   ├── evaluate.py              # metrics (aspect_f1, polarity_f1, per-aspect)
│   ├── train.py                 # train baseline
│   ├── train_fused.py           # train fused model + robustness eval
│   ├── evaluate_robustness.py   # compare any saved checkpoint's clean vs perturbed F1
│   └── infer.py                 # single-comment inference
└── checkpoints/                # created on first training run
```

---

## Step-by-step: how to test everything

### 0. Setup (once)

```bash
cd viet-absa
pip install -r requirements.txt
```

If you're on Colab, put this in the first cell after uploading/unzipping the
repo, and make sure the runtime is set to GPU (Runtime → Change runtime type
→ GPU).

### 1. Local sanity check — no GPU, no model download (~10 seconds)

```bash
python smoke_test.py
```

**Expect:** 4 checks, all `OK` — label parsing matches known rows, zero
label contradictions across the full train split, correct dataset tensor
shapes, and an aspect-frequency printout (you should see `STORAGE` around
91, `GENERAL` around 4,866 — that imbalance is expected and already handled
by class weighting).

```bash
python tests/test_perturb.py
```

**Expect:** `7/7 tests passed`. This checks the augmentation functions don't
crash on empty strings, single characters, or non-Vietnamese text, and that
`augment_fixed` is properly deterministic.

If either of these fails, stop here — nothing downstream will work correctly
either, and it's much cheaper to debug without a GPU running.

### 2. Train the baseline (do this first, GPU required)

```bash
python -m src.train --config config/default.yaml
```

This downloads `vinai/phobert-base` on first run (a few hundred MB), then
trains for up to 15 epochs with early stopping (stops if dev `polarity_f1`
hasn't improved in 4 epochs). Takes roughly 20–40 minutes on a Colab T4,
depending on how early it stops.

**What to watch during training:**
- `epoch N | loss ... | dev aspect_f1=... polarity_f1=...` printed every
  epoch — loss should trend down, polarity_f1 should trend up (not
  monotonically, but overall).
- The `per-aspect F1` line — check whether `STORAGE` is getting non-zero F1.
  If it's stuck at 0 across several epochs, the class weighting isn't
  helping enough and is worth flagging back to me.

**What you get at the end:**
```
TEST: {'aspect_f1': ..., 'polarity_f1': ..., 'per_aspect_f1': {...}}
manifest written to checkpoints/baseline_manifest.json
```

Compare `TEST['polarity_f1']` and `TEST['aspect_f1']` against the ~78.76%/
~86.03% target. Reasonably close (within a few points) is a legitimate
reproduction; exact match isn't required or expected on a first pass.

**If you want to iterate faster while debugging** (e.g. checking the loop
runs without crashing before committing to a full run):
```bash
python -m src.train --config config/default.yaml --epochs 1 --batch 16
```

### 3. Train the fused model (only after step 2 looks reasonable)

```bash
python -m src.train_fused --config config/default.yaml
```

Same training loop as the baseline, plus:
- `char_ids` (raw byte-level input) alongside the usual PhoBERT input
- Adversarial augmentation applied to the train split only (50% of samples,
  1–2 random perturbations each — see `src/data/perturb.py`)
- At the end, evaluates on both a clean test set and a **fixed, reproducible**
  perturbed copy of test, and reports:

```
TEST (clean):     {...}
TEST (perturbed): {...}
ROBUSTNESS DELTA (polarity_f1, clean - perturbed): X.XX
```

Lower delta = more robust to obfuscated/typo'd text. This delta is the
number you want to be smaller for the fused model than for the baseline —
that comparison (next step) is the actual contribution claim.

### 4. Compare robustness: baseline vs. fused

Once both checkpoints exist:

```bash
python -m src.evaluate_robustness --model baseline --checkpoint checkpoints/baseline.pt
python -m src.evaluate_robustness --model fused    --checkpoint checkpoints/fused.pt
```

Put both deltas side by side. If the fused model's delta is meaningfully
smaller, that's your robustness result. If it's about the same or worse,
that's also a real (and reportable) finding — don't force a positive result
if the data doesn't support it.

### 5. Spot-check predictions qualitatively

```bash
python -m src.infer --model fused --checkpoint checkpoints/fused.pt \
    --text "Pin trâu, camera đẹp nhưng giá hơi cao"
```

**Expect** something like:
```
Text: Pin trâu, camera đẹp nhưng giá hơi cao
  BATTERY      -> Positive
  CAMERA       -> Positive
  PRICE        -> Negative
```
(aspects not mentioned are omitted). Try a few sentences with obvious
sentiment to eyeball whether predictions look sane before trusting the
aggregate metrics.

### 6. After each run, check the manifest

```bash
cat checkpoints/baseline_manifest.json
cat checkpoints/fused_manifest.json
```

Each contains the exact args used and the full per-epoch history — this is
your source of truth for any table or plot in the paper, so you don't need
to re-scrape terminal output.

---

## Troubleshooting

- **`CUDA out of memory`** — lower `--batch` (try 16 or 8) via CLI override,
  e.g. `python -m src.train --config config/default.yaml --batch 16`.
- **Training loss is `nan`** — usually a learning-rate or class-weight
  instability issue; try `--no_class_weights` first to isolate whether the
  weighting (capped at 20x) is the cause.
- **`STORAGE` F1 stuck at 0 across all epochs** — the class weighting may
  not be enough on its own given only 91 training examples; worth flagging
  back rather than treating it as a bug in the code.
- **Fused model trains much slower than baseline** — expected; char-CNN adds
  compute, and `include_chars=True` datasets do extra byte-encoding per
  sample. If it's *dramatically* slower (not just ~10-20% more), check that
  AMP is actually enabling (`--no_amp` should NOT be set).
