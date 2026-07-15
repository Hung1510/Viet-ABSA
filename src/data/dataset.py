"""UIT-ViSFD multi-task ABSA torch Dataset."""
import csv
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset

from .preprocess import preprocess as default_preprocess, normalize
from .labels import parse_label, NUM_ASPECTS, NUM_STATES
from ..models.char_cnn import text_to_byte_ids


class ViSFDDataset(Dataset):
    """Each item -> dict(input_ids, attention_mask, labels[, char_ids]).

    labels: LongTensor [NUM_ASPECTS], each in {0:none, 1:pos, 2:neu, 3:neg}.

    Pass a `preprocess_fn` to inject perturbations later (e.g. adversarial
    augmentation) without touching this class.

    Set `include_chars=True` to also emit `char_ids` (raw UTF-8 byte ids of
    the *unsegmented* text -- see models/char_cnn.py for why) for the fused
    PhoBERT+char-CNN model. Baseline usage (train.py) leaves this False and
    is unaffected by any of this.

    `augment_fn`, if given, is applied to the RAW comment text before
    normalization/segmentation (so typo/tone-drop/teencode perturbations act
    on the true surface form) -- pass this only for the TRAIN split; dev/test
    should stay clean, or use perturb.augment_fixed() to build a separate,
    reproducible perturbed eval copy instead.
    """

    def __init__(self, csv_path: str, tokenizer, max_len: int = 256,
                 preprocess_fn: Optional[Callable[[str], str]] = None,
                 include_chars: bool = False, max_chars: int = 300,
                 augment_fn: Optional[Callable[[str], str]] = None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pre = preprocess_fn or default_preprocess
        self.include_chars = include_chars
        self.max_chars = max_chars
        self.augment_fn = augment_fn
        self.samples = []
        with open(csv_path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                self.samples.append((row["comment"], parse_label(row["label"])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        text, states = self.samples[i]
        if self.augment_fn is not None:
            text = self.augment_fn(text)

        enc = self.tokenizer(
            self.pre(text),
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(states, dtype=torch.long),  # [NUM_ASPECTS]
        }
        if self.include_chars:
            # unsegmented (normalize-only) text, per char_cnn.py's rationale
            byte_ids = text_to_byte_ids(normalize(text), max_chars=self.max_chars)
            item["char_ids"] = torch.tensor(byte_ids, dtype=torch.long)
        return item


def compute_class_weights(csv_path: str, cap: float = 20.0) -> torch.Tensor:
    """Per-aspect inverse-frequency class weights, shape [NUM_ASPECTS, NUM_STATES].

    Motivation: aspect frequency is heavily skewed (e.g. in Train.csv, GENERAL
    appears ~4,866x vs STORAGE ~91x). Unweighted CE lets the model ignore rare
    aspects/states almost entirely since their contribution to total loss is
    tiny. Weight w[a, s] = N_a / (NUM_STATES * count[a, s]), clipped to `cap`
    so a near-empty cell (e.g. a state with 1 example) doesn't blow up the
    loss scale and destabilize training.

    Pass the result to ViSFDBaseline.loss(..., class_weights=...).
    """
    counts = torch.zeros(NUM_ASPECTS, NUM_STATES)
    n = 0
    with open(csv_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            states = parse_label(row["label"])
            for a, s in enumerate(states):
                counts[a, s] += 1
            n += 1
    weights = n / (NUM_STATES * counts.clamp(min=1))
    return weights.clamp(max=cap)
