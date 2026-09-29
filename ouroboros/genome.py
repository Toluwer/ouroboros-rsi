"""The genome: self-modifiable training parameters under hard bounds.

This is Ouroboros's bounded self-modification mechanism. The daemon may
change its own sampling temperature, number of attempts per question,
training steps, learning rate, and so on -- but only:

  1. within the hard bounds declared in config/constitution.json,
  2. one experiment at a time,
  3. on a fixed cadence (constitution.mutation_every),
  4. with the change retained only if the next evaluation gate accepts the
     cycle; otherwise the previous value is restored.

This is a deliberately conservative form of self-modification: the system
tunes its own learning process, not its own code, its task, its evaluation,
or its safety envelope. Those are constitutional invariants.
"""

from __future__ import annotations

import random

from . import config

_CONTINUOUS = {"temperature", "lr"}
_INT_KEYS = {"k", "max_new_tokens", "gen_questions", "rationalize_cap",
             "replay", "train_steps", "cooldown_s"}


def _bounds(constitution: dict, key: str) -> tuple[float, float]:
    lo, hi = constitution["bounds"][key]
    return float(lo), float(hi)


def clamp(key: str, value, constitution: dict):
    lo, hi = _bounds(constitution, key)
    if key in _CONTINUOUS:
        v = round(min(max(float(value), lo), hi), 6)
        return v
    return int(min(max(int(round(float(value))), int(lo)), int(hi)))


def maybe_mutate(genome: dict, state: dict, constitution: dict, rng: random.Random) -> bool:
    """Start a genome experiment if cadence allows and none is pending.

    The new value takes effect for the current cycle; the pending record is
    settled after the evaluation gate by settle().
    """
    if state.get("pending_mutation"):
        return False
    cadence = int(constitution.get("mutation_every", 5))
    next_cycle = state["cycle"] + 1
    if next_cycle % cadence != 0:
        return False
    keys = list(constitution["bounds"].keys())
    key = rng.choice(keys)
    lo, hi = _bounds(constitution, key)
    if key in _CONTINUOUS:
        # sample on a log scale between bounds
        import math

        new = math.exp(rng.uniform(math.log(lo), math.log(hi)))
    else:
        new = rng.randint(int(lo), int(hi))
    new = clamp(key, new, constitution)
    old = genome.get(key)
    if new == old:
        return False
    genome[key] = new
    config.save_genome(genome)
    state["pending_mutation"] = {"key": key, "old": old, "new": new,
                                 "cycle": next_cycle}
    return True


def settle(genome: dict, state: dict, accepted: bool) -> dict | None:
    """Resolve a pending experiment after the evaluation gate."""
    pm = state.get("pending_mutation")
    if not pm:
        return None
    if accepted:
        state.setdefault("mutated_keys", []).append(
            {"key": pm["key"], "old": pm["old"], "new": pm["new"],
             "cycle": pm["cycle"], "retained": True}
        )
        state["pending_mutation"] = None
        return pm
    genome[pm["key"]] = pm["old"]
    config.save_genome(genome)
    state["pending_mutation"] = None
    return pm
