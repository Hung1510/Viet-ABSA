"""Text normalization + word segmentation for PhoBERT.

PhoBERT expects *word-segmented* input (multi-syllable words joined by "_").
Default segmenter: underthesea (pip, no Java). Paper-faithful alternative:
VnCoreNLP via py_vncorenlp (needs Java) — swap in `segment` if you want to
match published numbers exactly.
"""
import re
import unicodedata

try:
    from underthesea import word_tokenize as _uts
    _HAS_UTS = True
except Exception:
    _HAS_UTS = False

_WS = re.compile(r"\s+")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def normalize(text: str) -> str:
    """Conservative cleanup. Keeps case (PhoBERT-base is cased) and emojis
    (they carry sentiment). NFC unifies Vietnamese diacritic encodings."""
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\\n", " ")   # stray literal backslash-n, if any
    text = _CTRL.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text


def segment(text: str) -> str:
    """Join multi-syllable words with '_' for PhoBERT. Falls back to raw
    syllables if underthesea isn't installed (runs, slightly lower F1)."""
    if _HAS_UTS:
        return _uts(text, format="text")
    return text


def preprocess(text: str) -> str:
    return segment(normalize(text))
