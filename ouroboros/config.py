"""Central path and configuration handling for Ouroboros.

All runtime artifacts (checkpoints, curated data, evaluation ledgers, logs)
live under workspace/ so the autonomous daemon owns its own state without
writing into the source tree. Source-of-truth configuration is split in two:

  config/genome.json       -- mutable training hyperparameters. The daemon
                              itself may modify these, but only within the
                              bounds defined by the constitution, and only
                              keeping mutations that survive evaluation.
  config/constitution.json -- immutable invariants and hard bounds. Written
                              by humans; the daemon must never modify it.

State (cycle counter, best checkpoint, evaluation frontier) is persisted in
workspace/state.json with atomic writes so a crash at any point leaves the
system in a recoverable state.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"
WORKSPACE = REPO_ROOT / "workspace"

W_GENERATIONS = WORKSPACE / "generations"
W_CURATED = WORKSPACE / "curated"
W_CHECKPOINTS = WORKSPACE / "checkpoints"
W_EVALS = WORKSPACE / "evals"
W_LOGS = WORKSPACE / "logs"
ALL_WS_DIRS = [WORKSPACE, W_GENERATIONS, W_CURATED, W_CHECKPOINTS, W_EVALS, W_LOGS]

# Keep every cache inside the project tree: portable, inspectable, and the
# daemon can keep improving with no network access once data is cached.
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=False)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load_genome() -> dict:
    return _read_json(CONFIG_DIR / "genome.json")


def save_genome(genome: dict) -> None:
    _atomic_write_json(CONFIG_DIR / "genome.json", genome)


def load_constitution() -> dict:
    return _read_json(CONFIG_DIR / "constitution.json")


def load_remote() -> dict:
    """Committed, secret-free remote description: {owner, repo}."""
    p = CONFIG_DIR / "remote.json"
    return _read_json(p) if p.exists() else {"owner": "", "repo": ""}


def ensure_workspace() -> None:
    for d in ALL_WS_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def state_path() -> Path:
    return WORKSPACE / "state.json"


def default_state() -> dict:
    return {
        "cycle": 0,                 # number of completed improvement cycles
        "best_acc": None,           # held-out accuracy of best checkpoint
        "best_ckpt": None,          # path to best checkpoint
        "ptr": 0,                   # rotating pointer into the train pool
        "pending_mutation": None,   # active genome experiment, if any
        "mutated_keys": [],         # history of retained mutations
        "started": None,
        "last_cycle_at": None,
    }


def load_state() -> dict:
    p = state_path()
    if p.exists():
        st = _read_json(p)
    else:
        st = default_state()
    base = default_state()
    base.update(st)
    return base


def save_state(state: dict) -> None:
    _atomic_write_json(state_path(), state)


def ledger_path() -> Path:
    return W_EVALS / "ledger.jsonl"


def append_ledger(entry: dict) -> None:
    ledger_path().parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=False) + "\n")


def evolution_log_path() -> Path:
    return REPO_ROOT / "EVOLUTION.md"


def append_evolution(line: str) -> None:
    with open(evolution_log_path(), "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")
