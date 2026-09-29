"""Phase 0: operator-run supervised fine-tuning.

The system is never allowed to act autonomously before this step completes.
Phase 0 fine-tunes the base model on real CommonsenseQA training data
(direct-answer format), records the held-out baseline, and freezes the
initial checkpoint that the improvement loop bootstraps from.

Training is resumable: progress is checkpointed every chunk (see
PHASE0["chunk"]), so the run survives interruption and continues to the
configured step target. Repeated invocation is idempotent -- once phase 0
is complete, subsequent calls return the recorded entry without retraining.
"""

from __future__ import annotations

import json
import time

from . import config, data, evaluate
from .model import OuroborosModel

PHASE0 = {
    "train_examples": 3000,   # real CSQA training questions
    "steps": 720,             # 2 epochs at batch 8
    "lr": 6e-5,
    "batch_size": 8,
    "chunk": 120,             # steps between checkpoints
}
PHASE0_FAST = {
    "train_examples": 400,
    "steps": 60,
    "lr": 3e-5,
    "batch_size": 8,
    "chunk": 30,
}

_PARTIAL = config.W_CHECKPOINTS / "phase0-partial"
_PROGRESS = config.W_LOGS / "phase0_progress.json"


def _read_progress() -> dict:
    if _PROGRESS.exists():
        with open(_PROGRESS, encoding="utf-8") as f:
            return json.load(f)
    return {"steps_done": 0}


def _write_progress(prog: dict) -> None:
    _PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with open(_PROGRESS, "w", encoding="utf-8") as f:
        json.dump(prog, f)


def run_phase0(fast: bool = False, budget_s: float | None = None, log=print) -> dict | None:
    config.ensure_workspace()
    constitution = config.load_constitution()
    params = dict(PHASE0_FAST if fast else PHASE0)

    if (config.W_EVALS / "phase0.json").exists() and not _PARTIAL.exists():
        state = config.load_state()
        if state.get("best_ckpt"):
            log("[phase0] already complete; returning recorded entry")
            with open(config.W_EVALS / "phase0.json", encoding="utf-8") as f:
                return json.load(f)

    base_model = constitution["immutable"]["base_model"]
    eval_n = 60 if fast else constitution["immutable"]["eval_n"]
    eval_seed = constitution["immutable"]["eval_seed"]

    prog = _read_progress()
    log(f"[phase0] base model: {base_model} | resume at step {prog['steps_done']}/{params['steps']}")

    items = data.train_items()
    train = items[: params["train_examples"]]
    examples = [(data.direct_prompt(it), data.direct_completion(it)) for it in train]

    if prog["steps_done"] == 0:
        model = OuroborosModel(base_model)
    else:
        model = OuroborosModel.load(str(_PARTIAL))

    t0 = time.time()
    while prog["steps_done"] < params["steps"]:
        if budget_s is not None and (time.time() - t0) > budget_s - 90:
            log(f"[phase0] budget reached at step {prog['steps_done']}; "
                f"checkpoint saved, rerun to continue")
            model.save(str(_PARTIAL))
            _write_progress(prog)
            return None
        n = min(params["chunk"], params["steps"] - prog["steps_done"])
        log(f"[phase0] training chunk: {n} steps "
            f"({prog['steps_done']}->{prog['steps_done'] + n})")
        model.fine_tune(
            examples,
            steps=n,
            lr=params["lr"],
            batch_size=params["batch_size"],
            seed=42 + prog["steps_done"],
            log=log,
        )
        prog["steps_done"] += n
        model.save(str(_PARTIAL))
        _write_progress(prog)

    # ---- completion: held-out baseline ---------------------------------
    ckpt = config.W_CHECKPOINTS / "ouroboros-v0"
    import os
    import shutil

    if _PARTIAL.exists():
        if ckpt.exists():
            shutil.rmtree(ckpt)
        os.rename(_PARTIAL, ckpt)
    else:
        model.save(str(ckpt))
    _PROGRESS.unlink(missing_ok=True)
    log(f"[phase0] checkpoint finalized: {ckpt}")

    log(f"[phase0] held-out evaluation on {eval_n} validation questions")
    ev = evaluate.evaluate(model, data.eval_subset(eval_n, eval_seed))
    ev.pop("details", None)
    log(f"[phase0] accuracy: {ev['accuracy']:.3f} ({ev['correct']}/{ev['n']}) "
        f"in {ev['seconds']}s")

    entry = {
        "cycle": 0,
        "phase": "phase0",
        "checkpoint": str(ckpt),
        "train": {
            "steps": params["steps"],
            "lr": params["lr"],
            "examples": params["train_examples"],
            "dataset": "tau/commonsense_qa train split",
        },
        "eval": ev,
        "accepted": True,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    config.append_ledger(entry)
    with open(config.W_EVALS / "phase0.json", "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2)

    state = config.default_state()
    state.update(
        {
            "cycle": 0,
            "best_acc": ev["accuracy"],
            "best_ckpt": str(ckpt),
            "started": entry["timestamp"],
            "last_cycle_at": entry["timestamp"],
        }
    )
    config.save_state(state)

    config.append_evolution(
        f"| 0 | phase 0 (operator SFT on CommonsenseQA train) | "
        f"{ev['accuracy']:.3f} | accepted | baseline checkpoint `ouroboros-v0` |"
    )
    log("[phase0] done. State initialized; the daemon may now take over.")
    return entry
