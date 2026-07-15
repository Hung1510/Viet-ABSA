"""Adversarial-perturbation augmentation for Vietnamese ABSA robustness.

Three families, matched to how obfuscation actually shows up in Vietnamese
e-commerce comments:
  - typo       : keyboard-adjacent substitution / deletion / transposition
  - tone_drop  : strips diacritics -- "không đẹp" -> "khong dep". Extremely
                 common in fast casual typing.
  - teencode   : common informal abbreviations -- "không"->"ko", "được"->"dc"

IMPORTANT CAVEAT (flagging honestly rather than overclaiming authenticity):
`_ADJACENT` below is a rough QWERTY-adjacency table, not a validated Telex/VNI
keyboard-error model, and `_TEENCODE` is a small hand-picked dictionary, not
mined from real teencode usage. Both are reasonable starting proxies for
"does the model degrade gracefully under surface-form noise" -- they are NOT
a substitute for auditing a sample of real obfuscated comments from your data
and checking these transformations actually resemble what you see. Do that
audit before writing the robustness section of the paper; cite it as your
own perturbation model, not as a linguistically validated one.

Usage: apply only to the TRAIN split (via ViSFDDataset's augment_fn param in
train_fused.py). Never perturb dev/test with `augment()` (randomized) --
for the robustness-delta *evaluation*, build a separate, fixed perturbed copy
of test (see `augment_fixed` / evaluate_robustness in train_fused.py) so the
comparison is reproducible across runs.
"""
import random
import re
import unicodedata

_ADJACENT = {  # rough QWERTY adjacency over ASCII base letters (see caveat above)
    'a': 'qsz', 's': 'awedxz', 'd': 'serfcx', 'f': 'drtgvc', 'g': 'ftyhbv',
    'h': 'gyujnb', 'j': 'huikmn', 'k': 'jiolm', 'l': 'kop',
    'q': 'wa', 'w': 'qeas', 'e': 'wrds', 'r': 'etdf', 't': 'ryfg',
    'y': 'tugh', 'u': 'yihj', 'i': 'uojk', 'o': 'ipkl', 'p': 'ol',
    'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn',
    'n': 'bhjm', 'm': 'njk',
}

_TEENCODE = {  # small hand-picked set -- audit against real data, see caveat
    "không": ["ko", "k", "hong", "khum"],
    "được": ["dc", "đc", "dk"],
    "biết": ["bit", "bt"],
    "rất": ["r"],
    "vậy": ["v", "z"],
    "gì": ["j"],
    "giờ": ["h"],
    "một": ["1"],
    "hai": ["2"],
    "yêu": ["iu"],
    "anh": ["a"],
    "em": ["e"],
    "quá": ["wa", "qa"],
    "trời": ["troi"],
}
_TEENCODE_RE = re.compile(
    "|".join(re.escape(k) for k in sorted(_TEENCODE, key=len, reverse=True)),
    re.IGNORECASE,
)


def tone_drop(text: str) -> str:
    """Strip Vietnamese tone/vowel diacritics via NFD decomposition + combining-
    mark removal; 'đ'/'Đ' handled separately since it's not a combining case."""
    nfd = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in nfd if not unicodedata.combining(ch))
    stripped = stripped.replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFC", stripped)


def teencode(text: str) -> str:
    def repl(m):
        word = m.group(0).lower()
        return random.choice(_TEENCODE.get(word, [word]))
    return _TEENCODE_RE.sub(repl, text)


def typo(text: str, rate: float = 0.06) -> str:
    """Per-character random substitution/deletion/adjacent-transposition at
    `rate`, restricted to alphabetic ASCII-adjacency keys; punctuation and
    Vietnamese-only diacritic characters are left untouched (this only
    simulates Latin-keyboard slips, not diacritic mistyping -- see caveat)."""
    chars = list(text)
    out = []
    i = 0
    while i < len(chars):
        ch = chars[i]
        if ch.isalpha() and ch.lower() in _ADJACENT and random.random() < rate:
            op = random.choice(["sub", "del", "swap"])
            if op == "sub":
                repl = random.choice(_ADJACENT[ch.lower()])
                out.append(repl.upper() if ch.isupper() else repl)
            elif op == "swap" and i + 1 < len(chars):
                out.append(chars[i + 1])
                out.append(ch)
                i += 1
            # "del": append nothing, character dropped
        else:
            out.append(ch)
        i += 1
    return "".join(out)


_ALL_OPS = [typo, tone_drop, teencode]


def augment(text: str, p: float = 0.5) -> str:
    """Randomized augmentation for TRAINING: with probability `p`, apply 1-2
    randomly chosen perturbation ops (not all 3, to keep some variety in how
    corrupted each sample is)."""
    if random.random() > p:
        return text
    for op in random.sample(_ALL_OPS, k=random.randint(1, 2)):
        text = op(text)
    return text


def augment_fixed(text: str, seed: int = 0) -> str:
    """Deterministic augmentation for building a reproducible perturbed
    EVAL set (robustness-delta measurement). Always applies all 3 ops in a
    fixed order so the same input always produces the same output,
    independent of global random state."""
    rng_state = random.getstate()
    random.seed(hash((text, seed)) & 0xFFFFFFFF)
    out = teencode(tone_drop(typo(text, rate=0.10)))
    random.setstate(rng_state)
    return out
