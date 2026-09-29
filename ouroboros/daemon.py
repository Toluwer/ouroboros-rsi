"""The autonomous daemon.

Runs improvement cycles forever: sample -> filter -> fine-tune -> gate ->
publish. Crash-safe (a failing cycle is logged and retried, never kills the
daemon), offline-tolerant (pushes queue up locally until connectivity
returns), and rate-limited by genome.cooldown_s.

Launch:
    GITHUB_TOKEN=... nohup python -m ouroboros daemon \
        >> workspace/logs/daemon.out 2>&1 &
"""

from __future__ import annotations

import json
import time
import traceback

from . import config
from .cycle import run_cycle


def run(fast: bool = False) -> None:
    config.ensure_workspace()
    state = config.load_state()
    if not state.get("best_ckpt"):
        raise SystemExit("No incumbent checkpoint found. Run `python -m ouroboros phase0` first.")

    print(f"ouroboros daemon: starting from cycle {state['cycle']} "
          f"(best_acc={state['best_acc']})", flush=True)
    while True:
        t0 = time.time()
        try:
            genome = config.load_genome()
            entry = run_cycle(fast=fast)
            print(f"[daemon] cycle {entry['cycle']}: "
                  f"acc={entry['accuracy']:.3f} accepted={entry['accepted']} "
                  f"({round(time.time() - t0)}s)", flush=True)
        except Exception:
            err = traceback.format_exc()
            with open(config.W_LOGS / "errors.log", "a", encoding="utf-8") as f:
                f.write(f"--- {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n{err}\n")
            print(f"[daemon] cycle failed, logged to workspace/logs/errors.log:\n{err}",
                  flush=True)
            time.sleep(120)
            continue
        try:
            cooldown = config.load_genome()["cooldown_s"]
        except Exception:
            cooldown = 60
        elapsed = time.time() - t0
        time.sleep(max(5, cooldown - elapsed))


def status() -> dict:
    state = config.load_state()
    ledger = []
    p = config.ledger_path()
    if p.exists():
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    ledger.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return {"state": state, "ledger_tail": ledger[-10:]}
