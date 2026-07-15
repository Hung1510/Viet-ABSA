"""UIT-ViSFD label schema + parsing. No external deps."""
import re

ASPECTS = ["BATTERY", "CAMERA", "DESIGN", "FEATURES", "GENERAL",
           "PERFORMANCE", "PRICE", "SCREEN", "SER&ACC", "STORAGE"]
ASPECT_TO_IDX = {a: i for i, a in enumerate(ASPECTS)}

# Per-aspect state: 0 = aspect absent, else its polarity.
POLARITY_TO_ID = {"none": 0, "Positive": 1, "Neutral": 2, "Negative": 3}
ID_TO_POLARITY = {v: k for k, v in POLARITY_TO_ID.items()}

NUM_ASPECTS = len(ASPECTS)   # 10
NUM_STATES = 4               # none / pos / neu / neg

_PAIR = re.compile(r"\{([^}#]+)#([^}]+)\}")


def parse_label(label: str):
    """'{BATTERY#Negative};{GENERAL#Positive};{OTHERS};' -> [10] state ids.

    {OTHERS} and any aspect outside the canonical 10 are ignored (standard
    ViSFD 10-aspect ABSA setup). Returns a list of length NUM_ASPECTS.
    """
    states = [0] * NUM_ASPECTS
    for aspect, pol in _PAIR.findall(label or ""):
        aspect, pol = aspect.strip(), pol.strip()
        if aspect in ASPECT_TO_IDX and pol in POLARITY_TO_ID:
            states[ASPECT_TO_IDX[aspect]] = POLARITY_TO_ID[pol]
    return states
