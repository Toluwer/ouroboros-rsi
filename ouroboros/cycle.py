"""One self-improvement cycle (verified self-training), resumable.

The mechanism, per cycle, in constitution-declared ``selfgen_mode``:

  answer (default for GPT-2-class bases) -- verified self-training:
  1. SAMPLE     Draw the next batch of questions from the full 9,741-question
                pool (phase 0 consumed only the first 3,000, so most sampled
                questions were never seen in training).
  2. GENERATE   For each question, sample k answer completions at the
                genome's temperature.
  3. FILTER     Keep completions whose parsed answer matches ground truth
                (STaR's correctness filter). The model's own verified output
                strings become the training targets (self-distillation).
                For every question with no correct attempt, the gold
                completion is added instead (hard-example mining -- the
                rationalization step of STaR, minus free text, which bases
                of this scale cannot reliably produce).
  4. TRAIN      Fine-tune on (verified self-generated completions +
                hard-mined golds + replayed gold examples).
  5. GATE       Evaluate on the fixed held-out subset. Accept only if
                accuracy beats the incumbent by more than the constitution's
                margin; otherwise roll the weights back to the incumbent.
  6. PUBLISH    Append to the ledger and EVOLUTION.md, save artifacts, and
                commit+push to the repository (offline-tolerant).

  rationale (for instruction-following bases) -- STaR bootstrapping with
  few-shot rationale prompts; the same cycle code path, different
  generation/filtering stage, switched by the constitution.

The loop is recursive in the meaningful sense: the model's own competence
selects its curriculum (what it gets right is consolidated; what it misses
is hard-mined), the gate converts that selection into new weights only
when held-out accuracy actually improves, and the improved model then
re-selects the next cycle's curriculum.

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


def _gen_ok(text: str, mode: str) -> bool:
    words = len(text.split())
    return (1 <= words <= 8) if mode == "answer" else (8 <= words <= 90)


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
    # the few-shot exemplar questions are excluded from the generation pool
    # so their answers are never present in the sampling prompt
    exemplar_qs = {ex["question"] for ex in data.EXEMPLARS}
    pool = [it for it in items if it["question"] not in exemplar_qs]

    n_gen = 12 if fast else genome["gen_questions"]
    k = genome["k"]
    ptr = state["ptr"]
    batch = [pool[(ptr + i) % len(pool)] for i in range(n_gen)]
    state["ptr"] = (ptr + n_gen) % len(pool)
    config.save_state(state)

    t0 = time.time()
    mode = constitution["immutable"].get("selfgen_mode", "answer")
    positives: list[tuple[dict, str]] = []      # (item, model's own verified completion)
    hard_mined: list[dict] = []                 # misses -> gold completions
    raw_generations = []
    for it in batch:
        if mode == "rationale":
            prompt = data.fs_rationale_prompt(it)
            max_new = genome["max_new_tokens"]
        else:
            prompt = data.direct_prompt(it)
            max_new = 12
        gens = model.generate(
            prompt,
            k=k,
            temperature=genome["temperature"],
            max_new_tokens=max_new,
        )
        kept: list[str] = []
        for g in gens:
            if mode == "rationale":
                pred = data.parse_letter(g)
            else:
                pred = data.parse_prediction(g, it)
            if pred == it["answer"] and _gen_ok(g, mode) and g.strip().lower() not in kept:
                kept.append(g.strip().lower())
                positives.append((it, g.strip()))
                raw_generations.append(
                    {"q": it["question"][:100], "gold": it["answer"],
                     "gen": g.strip()[:200], "source": "verified-self"}
                )
        if not kept:
            hard_mined.append(it)
            raw_generations.append(
                {"q": it["question"][:100], "gold": it["answer"],
                 "gen": "(no correct attempt -- hard-mined with gold)",
                 "source": "hard-mined"}
            )
    gen_seconds = round(time.time() - t0, 1)
    log(f"[cycle {cycle_id}] sampled {len(batch)} questions x{k}: "
        f"{len(positives)} verified self-completions, {len(hard_mined)} "
        f"hard-mined ({gen_seconds}s)")

    # training set: verified self-generated completions (kept verbatim) +
    # hard-mined golds + gold replay for format stability and retention
    examples: list[list[str]] = []
    for it, gen in positives:
        if mode == "rationale":
            examples.append([data.rationale_prompt(it), f" {gen}"])
        else:
            examples.append([data.direct_prompt(it), gen])
    for it in hard_mined:
        examples.append([data.direct_prompt(it), data.direct_completion(it)])
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
            "hard_mined": len(hard_mined),
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


def _elapsed_gen_estimate(fast: bool, genome: dict, mode: str = "answer") -> float:
    if fast:
        return 90
    if mode == "rationale":
        return genome["gen_questions"] * 2.6 + 60
    return genome["gen_questions"] * 0.9 + 100


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
            mode = constitution["immutable"].get("selfgen_mode", "answer")
            need = _elapsed_gen_estimate(fast, genome, mode) + 120
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
            # next version number: highest existing + 1 (collision-free even
            # against historical naming quirks; v0 is the operator baseline)
            existing = list(config.W_CHECKPOINTS.glob("ouroboros-v*"))
            n_ver = max((_ver(p.name) for p in existing), default=0) + 1
            ckpt = config.W_CHECKPOINTS / f"ouroboros-v{n_ver}"
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
            "hard_mined": gs["hard_mined"],
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
                f"| {cycle_id} | self-train (verified {gs['positives']} + "
                f"hard-mined {gs['hard_mined']}, train {job['steps_done']} steps) | "
                f"{ev['accuracy']:.3f} | accepted | "
                f"`{Path(ckpt).name}` "
                f"({'+' + format(delta, '.3f') if delta is not None else 'baseline'}) |"
            )
        else:
            config.append_evolution(
                f"| {cycle_id} | self-train (verified {gs['positives']} + "
                f"hard-mined {gs['hard_mined']}, train {job['steps_done']} steps) | "
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
        if rem is None or rem < _elapsed_gen_estimate(
                fast, genome,
                constitution["immutable"].get("selfgen_mode", "answer")) + 180:
            break

    return summaries
