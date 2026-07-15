"""Metrics for UIT-ViSFD multi-task ABSA.

Reports the two headline numbers:
  aspect_f1   : macro-F1 of aspect *presence* (present vs none), over 10 aspects.
  polarity_f1 : macro-F1 of joint aspect#polarity detection, i.e. every
                (aspect, {pos,neu,neg}) pair treated as its own class.

NOTE: to land *exactly* on a published figure, match that paper's metric
definition/script — small choices (micro vs macro, whether 'none' counts)
shift the number. These definitions are the common ABSA ones.
"""
import torch
from sklearn.metrics import f1_score, confusion_matrix

from .data.labels import NUM_ASPECTS, ASPECTS, ID_TO_POLARITY


@torch.no_grad()
def collect_preds(model, loader, device):
    model.eval()
    P, Y = [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        logits = model(ids, mask)                       # list of [B,4]
        preds = torch.stack([l.argmax(-1) for l in logits], dim=1)  # [B,10]
        P.append(preds.cpu())
        Y.append(batch["labels"])
    return torch.cat(P), torch.cat(Y)                   # [N,10], [N,10]


def compute_metrics(preds, labels):
    # aspect presence: state>0
    yt_asp = (labels > 0).reshape(-1).int().numpy()
    yp_asp = (preds > 0).reshape(-1).int().numpy()
    aspect_f1 = f1_score(yt_asp, yp_asp, average="macro", zero_division=0)

    # joint aspect#polarity: encode (aspect_idx, state) -> class id; 0 stays 0 (none)
    def encode(t):
        out = t.clone()
        for a in range(NUM_ASPECTS):
            col = out[:, a]
            out[:, a] = torch.where(col > 0, col + a * 3, torch.zeros_like(col))
        return out.reshape(-1).numpy()
    pol_f1 = f1_score(encode(labels), encode(preds), average="macro", zero_division=0)

    # per-aspect polarity F1 (state 0..3, macro over the 4 states within that
    # aspect) so rare aspects (e.g. STORAGE) don't hide inside one aggregate
    # number -- use this to check whether class weighting is actually helping
    # the aspects that need it, rather than just moving the average around.
    per_aspect = {}
    for a in range(NUM_ASPECTS):
        f1 = f1_score(labels[:, a].numpy(), preds[:, a].numpy(),
                       average="macro", zero_division=0)
        per_aspect[ASPECTS[a]] = round(f1 * 100, 2)

    return {"aspect_f1": round(aspect_f1 * 100, 2),
            "polarity_f1": round(pol_f1 * 100, 2),
            "per_aspect_f1": per_aspect}


def plot_confusion_matrix(preds, labels, out_path, aspect=None):
    """Save a confusion matrix over the 4 polarity states (none/pos/neu/neg).

    If `aspect` (an ASPECTS name) is given, restricts to that single aspect's
    column -- use this for aspects flagged by analyze_errors.py, e.g. to
    check whether STORAGE errors skew toward over-predicting "negative".
    Otherwise pools all 10 aspect columns into one matrix.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    state_names = [ID_TO_POLARITY[i] for i in range(4)]
    if aspect is not None:
        a = ASPECTS.index(aspect)
        y_true = labels[:, a].numpy()
        y_pred = preds[:, a].numpy()
        title = f"Confusion matrix -- {aspect}"
    else:
        y_true = labels.reshape(-1).numpy()
        y_pred = preds.reshape(-1).numpy()
        title = "Confusion matrix -- all aspects pooled"

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4)); ax.set_xticklabels(state_names, rotation=45)
    ax.set_yticks(range(4)); ax.set_yticklabels(state_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return cm
