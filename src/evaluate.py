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
from sklearn.metrics import f1_score

from .data.labels import NUM_ASPECTS, ASPECTS


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
