"""One self-improvement cycle (STaR-style bootstrapping), resumable.

Per cycle, using only unlabeled questions as prompts:

  1. SAMPLE     Draw the next batch of training questions (rotating pointer,
                so the system works through the full 9,741-question pool).
  2. GENERATE   For each question, sample k rationale+answer completions.
  3. FILTER     Keep completions whose final answer matches ground truth
                (deduplicated, length-bounded). For questions the model
                missed, STaR *rationalization*: show the gold answer as a
                hint, keep the rationale the model produces for it. Labels
                are used only for filtering, never as generation input.
  4. TRAIN      Fine-tune on (filtered self-generated data + replayed gold
                direct-format examples to prevent format drift and
                forgetting).
  5. GATE       Evaluate on the fixed held-out subset. Accept only if
                accuracy beats the incumbent by more than the constitution's
                margin; otherwise roll the weights back to the incumbent.
  6. PUBLISH    Append to the ledger and EVOLUTION.md, save artifacts, and
                commit+push to the repository (offline-tolerant).

Execution is staged and resumable: the cycle's state (curated training
data, step counter, stage) is journaled to workspace/logs/job_<id>.json and
the in-training weights to workspace/checkpoints/cycle_<id>_partial. A
cycle therefore tolerates interruption at any point and continues where it
left off on the next invocation. With ``budget_s`` unset (daemon mode) a
call to run_cycle completes one full cycle; with a budget set, it advances
as far as the budget allows and returns the completed cycles.

The cycle counter increments on every attempt; the *version* number only
advances on accepted cycles, because versions correspond to checkpoints
that empirically beat their predecessor.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from pathlib import Path

import torch

from . import config, data, evaluate, genome as genome_mod, gitops
from .model import OuroborosModel

_SECONDS_PER_STEP = 3.5  # measured on 2 vCPU + 15% margin


def _sample_ok(rationale: str) -> bool:
    words = len(rationale.split())
    return 8 <= words <= 90


def _ver(name: str) -> int:
    try:
        return int(name.rsplit("v", 1)[1])
    except (ValueError, IndexError):
        return 0


def _job_path(cycle_id: int) -> Path:
    return config.W_LOGS / f"job_{cycle_id}.json"


def _partial_path(cycle_id: int) -> Path:
    return config.W_CHECKPOINTS / f"cycle_{cycle_id}_partial"


def _prune_checkpoints(keep: int) -> None:
    """Retain the newest `keep` checkpoints; never delete the operator
    baseline (v0) or the current best."""
    best = config.load_state().get("best_ckpt")
    protected = {"ouroboros-v0"}
    if best:
        protected.add(Path(best).name)
    ckpts = sorted(config.W_CHECKPOINTS.glob("ouroboros-v*"), key=lambda p: _ver(p.name))
    candidates = [p for p in ckpts if p.name not in protected]
    for p in candidates[:-keep] if keep > 0 else candidates:
        shutil.rmtree(p, ignore_errors=True)


def _init_job(cycle_id: int, state: dict, genome: dict, constitution: dict,
              fast: bool, log) -> dict:
    """Stage 1-3: generate, filter, rationalize, and journal the cycle."""
    imm = constitution["immutable"]
    torch.manual_seed(1000 + cycle_id)
    rng = random.Random(1000 + cycle_id)

    mutated = genome_mod.maybe_mutate(genome, state, constitution, rng)
    if mutated:
        pm = state["pending_mutation"]
        log(f"[cycle {cycle_id}] genome experiment: {pm['key']} "
            f"{pm['old']} -> {pm['new']}")
        config.save_state(state)

    model = OuroborosModel.load(state["best_ckpt"])
    items = data.train_items()

    n_gen = 12 if fast else genome["gen_questions"]
    k = genome["k"]
    ptr = state["ptr"]
    batch = [items[(ptr + i) % len(items)] for i in range(n_gen)]
    state["ptr"] = (ptr + n_gen) % len(items)
    config.save_state(state)

    t0 = time.time()
    positives: list[tuple[dict, str]] = []
    rationalized: list[tuple[dict, str]] = []
    raw_generations = []
    for it in batch:
        gens = model.generate(
            data.rationale_prompt(it),
            k=k,
            temperature=genome["temperature"],
            max_new_tokens=genome["max_new_tokens"],
        )
        kept: list[str] = []
        for g in gens:
            pred = data.parse_letter(g)
            if pred == it["answer"] and _sample_ok(g) and g.strip().lower() not in kept:
                kept.append(g.strip().lower())
                positives.append((it, g.strip()))
                raw_generations.append(
                    {"q": it["question"][:100], "gold": it["answer"],
                     "rationale": g.strip()[:400], "source": "sampled"}
                )
        if not kept and len(rationalized) < genome["rationalize_cap"]:
            r = model.generate(
                data.rationalization_prompt(it, it["answer"]),
                k=1,
                temperature=0.5,
                max_new_tokens=genome["max_new_tokens"],
            )[0]
            if _sample_ok(r):
                rationalized.append((it, r.strip()))
                raw_generations.append(
                    {"q": it["question"][:100], "gold": it["answer"],
                     "rationale": r.strip()[:400], "source": "rationalized"}
                )
    gen_seconds = round(time.time() - t0, 1)
    log(f"[cycle {cycle_id}] generated on {len(batch)} questions: "
        f"{len(positives)} correct rationales, {len(rationalized)} rationalized "
        f"({gen_seconds}s)")

    # training set: self-generated rationale format + gold replay
    examples: list[list[str]] = []
    for it, r in positives + rationalized:
        examples.append([data.rationale_prompt(it), data.rationale_completion(it, r)])
    replay = 40 if fast else genome["replay"]
    replay_items = rng.sample(items[:4000], min(replay, 4000))
    for it in replay_items:
        examples.append([data.direct_prompt(it), data.direct_completion(it)])

    job = {
        "cycle": cycle_id,
        "stage": "train",
        "steps_done": 0,
        "steps_target": 50 if fast else genome["train_steps"],
        "examples": examples,
        "gen_stats": {
            "questions": len(batch),
            "positives": len(positives),
            "rationalized": len(rationalized),
            "gen_seconds": gen_seconds,
            "raw": raw_generations,
        },
        "fast": fast,
    }
    with open(_job_path(cycle_id), "w", encoding="utf-8") as f:
        json.dump(job, f)
    return job


def _load_job(cycle_id: int) -> dict | None:
    p = _job_path(cycle_id)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_job(job: dict) -> None:
    with open(_job_path(job["cycle"]), "w", encoding="utf-8") as f:
        json.dump(job, f)


def _elapsed_gen_estimate(fast: bool, genome: dict) -> float:
    return 90 if fast else genome["gen_questions"] * 2.6 + 60


def run_cycle(fast: bool = False, budget_s: float | None = None, log=print) -> list[dict]:
    config.ensure_workspace()
    state = config.load_state()
    if not state.get("best_ckpt"):
        raise RuntimeError("No incumbent checkpoint. Run phase 0 first.")

    genome = config.load_genome()
    constitution = config.load_constitution()
    imm = constitution["immutable"]
    t0 = time.time()
    summaries: list[dict] = []

    def remaining() -> float | None:
        if budget_s is None:
            return None
        return budget_s - (time.time() - t0)

    while True:
        cycle_id = state["cycle"] + 1
        job = _load_job(cycle_id)

        if job is None:
            rem = remaining()
            need = _elapsed_gen_estimate(fast, genome) + 120
            if rem is not None and rem < need:
                log(f"[cycle {cycle_id}] not enough budget to start "
                    f"({rem:.0f}s < {need:.0f}s needed); stopping here")
                break
            job = _init_job(cycle_id, state, genome, constitution, fast, log)
            state = config.load_state()

        # ---- stage: train (resumable chunks) -------------------------
        if job["stage"] == "train":
            while job["steps_done"] < job["steps_target"]:
                rem = remaining()
                if rem is not None and rem < 150:
                    log(f"[cycle {cycle_id}] training at step {job['steps_done']}/"
                        f"{job['steps_target']}; budget stop, will resume")
                    break
                src = (_partial_path(cycle_id) if job["steps_done"] > 0
                       else state["best_ckpt"])
                model = OuroborosModel.load(str(src))
                want = job["steps_target"] - job["steps_done"]
                max_s = None if rem is None else max(60.0, rem - 90)
                log(f"[cycle {cycle_id}] training chunk from step "
                    f"{job['steps_done']} (target {job['steps_target']}, "
                    f"budget {f'{max_s:.0f}s' if max_s else 'unlimited'})")
                tlog = model.fine_tune(
                    [(p, c) for p, c in job["examples"]],
                    steps=want,
                    lr=genome["lr"],
                    batch_size=imm["batch_size"],
                    seed=1000 + cycle_id + job["steps_done"],
                    log=log,
                    max_seconds=max_s,
                )
                done = tlog["steps"]
                job["steps_done"] += done
                model.save(str(_partial_path(cycle_id)))
                _save_job(job)
                log(f"[cycle {cycle_id}] progress: {job['steps_done']}/"
                    f"{job['steps_target']} steps")
                if tlog["interrupted"] and done == 0:
                    break
            if job["steps_done"] < job["steps_target"]:
                break  # resume on next invocation

        # ---- stage: gate -----------------------------------------------
        model = OuroborosModel.load(str(_partial_path(cycle_id)))
        eval_n = 60 if fast else imm["eval_n"]
        ev = evaluate.evaluate(model, data.eval_subset(eval_n, imm["eval_seed"]))
        details = ev.pop("details")
        margin = imm["accept_margin"]
        incumbent = state["best_acc"]
        accepted = incumbent is None or ev["accuracy"] > incumbent + margin
        log(f"[cycle {cycle_id}] held-out accuracy {ev['accuracy']:.3f} vs incumbent "
            f"{incumbent if incumbent is None else round(incumbent, 3)} -> "
            f"{'ACCEPT' if accepted else 'ROLLBACK'}")

        gs = job["gen_stats"]
        train_log = {
            "steps": job["steps_done"],
            "lr": genome["lr"],
            "examples": len(job["examples"]),
        }

        ckpt = None
        if accepted:
            n_accepted = 1  # phase 0 counts as v0
            with open(config.ledger_path(), encoding="utf-8") as f:
                n_accepted += sum(1 for line in f if line.strip()
                                  and json.loads(line).get("accepted"))
            ckpt = config.W_CHECKPOINTS / f"ouroboros-v{n_accepted}"
            if _partial_path(cycle_id).exists():
                if ckpt.exists():
                    shutil.rmtree(ckpt)
                os.rename(_partial_path(cycle_id), ckpt)
            state["best_acc"] = ev["accuracy"]
            state["best_ckpt"] = str(ckpt)
            _prune_checkpoints(imm["max_checkpoints_retained"])
            ev["details"] = details
            with open(config.W_EVALS / f"cycle_{cycle_id}.json", "w",
                      encoding="utf-8") as f:
                json.dump({"cycle": cycle_id, "eval": ev, "train": train_log,
                           "generations": gs["raw"][:80]}, f, indent=1)
            ev.pop("details", None)
        else:
            shutil.rmtree(_partial_path(cycle_id), ignore_errors=True)

        pm = genome_mod.settle(genome, state, accepted)
        if pm:
            log(f"[cycle {cycle_id}] genome experiment {pm['key']}: "
                f"{'retained' if accepted else 'reverted'}")

        state["cycle"] = cycle_id
        state["last_cycle_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        config.save_state(state)
        _job_path(cycle_id).unlink(missing_ok=True)

        entry = {
            "cycle": cycle_id,
            "phase": "improve",
            "accepted": accepted,
            "accuracy": ev["accuracy"],
            "incumbent": incumbent,
            "checkpoint": str(ckpt) if ckpt else state["best_ckpt"],
            "positives": gs["positives"],
            "rationalized": gs["rationalized"],
            "gen_questions": gs["questions"],
            "train": train_log,
            "genome": {k: genome[k] for k in sorted(genome)},
            "mutation": pm,
            "timestamp": state["last_cycle_at"],
        }
        config.append_ledger(entry)

        if accepted:
            delta = (ev["accuracy"] - incumbent) if incumbent is not None else None
            config.append_evolution(
                f"| {cycle_id} | self-improve (sample {gs['positives']} + "
                f"rationalize {gs['rationalized']}, train {job['steps_done']} steps) | "
                f"{ev['accuracy']:.3f} | accepted | "
                f"`{Path(ckpt).name}` "
                f"({'+' + format(delta, '.3f') if delta is not None else 'baseline'}) |"
            )
        else:
            config.append_evolution(
                f"| {cycle_id} | self-improve (sample {gs['positives']} + "
                f"rationalize {gs['rationalized']}, train {job['steps_done']} steps) | "
                f"{ev['accuracy']:.3f} | rolled back | incumbent kept |"
            )

        sample_path = config.W_CURATED / f"sample_cycle_{cycle_id}.jsonl"
        with open(sample_path, "w", encoding="utf-8") as f:
            for row in gs["raw"][:60]:
                f.write(json.dumps(row) + "\n")
        with open(config.W_GENERATIONS / f"cycle_{cycle_id}.jsonl", "w",
                  encoding="utf-8") as f:
            for row in gs["raw"]:
                f.write(json.dumps(row) + "\n")

        push_paths = [
            "EVOLUTION.md",
            str(config.ledger_path()),
            str(config.state_path()),
            "config/genome.json",
            str(sample_path),
        ]
        if accepted:
            push_paths.append(f"workspace/evals/cycle_{cycle_id}.json")

        msg = (f"val_acc {ev['accuracy']:.3f} "
               f"({'accepted' if accepted else 'rolled back'}) "
               f"[cycle {cycle_id}, autonomous]")
        try:
            git = gitops.commit_and_push(f"cycle {cycle_id}", msg, push_paths)
            log(f"[cycle {cycle_id}] git: committed={git.get('committed')} "
                f"pushed={git.get('pushed')}")
            entry["git"] = git
        except RuntimeError as e:
            entry["git"] = {"error": str(e)}
            log(f"[cycle {cycle_id}] git error: {e}")

        summaries.append(entry)
        state = config.load_state()

        rem = remaining()
        if rem is None or rem < _elapsed_gen_estimate(fast, genome) + 180:
            break

    return summaries
