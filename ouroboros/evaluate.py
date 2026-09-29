"""Held-out evaluation.

Scoring always uses the direct-answer prompt format and greedy decoding, on
a fixed, cached validation subset (see data.eval_subset). This keeps
accuracy strictly comparable across phase 0 and every improvement cycle:
nothing about the scoring procedure changes as the system evolves.
"""

from __future__ import annotations

import time

from . import data


def evaluate(model, items: list[dict], max_new_tokens: int = 12) -> dict:
    t0 = time.time()
    correct = 0
    invalid = 0
    rows = []
    for it in items:
        text = model.generate(
            data.direct_prompt(it), k=1, max_new_tokens=max_new_tokens, do_sample=False
        )[0]
        pred = data.parse_prediction(text, it)
        ok = pred == it["answer"]
        if pred is None:
            invalid += 1
        if ok:
            correct += 1
        rows.append({"q": it["question"][:80], "gold": it["answer"], "pred": pred,
                     "gen": text.strip()[:12], "ok": ok})
    n = len(items)
    return {
        "accuracy": round(correct / max(1, n), 4),
        "correct": correct,
        "n": n,
        "invalid_predictions": invalid,
        "seconds": round(time.time() - t0, 1),
        "details": rows,
    }
