# Ouroboros

A self-improving language model system. A GPT-2 class base is fine-tuned by
an operator on a real benchmark (Phase 0), then handed to an autonomous
loop that generates its own training data, filters it against ground truth,
fine-tunes itself, and keeps the new weights only if held-out accuracy
actually improves. Every accepted cycle becomes a new version; every
rejected cycle is rolled back and recorded. The system tunes its own
hyperparameters within a human-written constitution, and runs offline.

**Repository state:** the system is live. `EVOLUTION.md` is its append-only
record; `workspace/evals/ledger.jsonl` is the machine-readable ledger;
commits marked `[cycle N, autonomous]` were made by the system itself.

## What this is, and what it is not

This is a working, end-to-end implementation of empirically gated
self-improvement: self-generated, ground-truth-verified training data; a
hard acceptance gate on a fixed held-out set; bounded self-modification of
its own training configuration; and autonomous publication of its results.
The loop is real — no step is simulated, and rejected cycles are genuinely
rolled back.

It is not, and does not claim to be, open-ended recursive self-improvement
in the strong sense. The system cannot rewrite its own code, change its
task, or alter its evaluation: those are constitutional invariants (see
`SAFETY.md`). Its improvement curve is bounded by the base model's capacity
and by the benchmark's structure. "Self-improvement" here has a precise,
measurable meaning: cycles whose held-out accuracy beats the incumbent
become the incumbent.

## Method

The loop derives from STaR (Zelikman et al., 2022, *STaR: Bootstrapping
Reasoning With Reasoning*) and the verified self-training lineage
(self-distillation with ground-truth filtering; cf. Noisy Student,
Xie et al. 2020). The constitution declares a `selfgen_mode`:

- **`answer`** (default, GPT-2 class bases): each cycle samples the next
  batch of questions from the 9,741-question CommonsenseQA train pool
  (Phase 0 consumed only the first 3,000, so most sampled questions are
  fresh), draws k sampled completions per question, and keeps completions
  whose parsed answer matches ground truth — the model's own verified
  output strings become training targets. Questions with no correct attempt
  are hard-mined: their gold completions join the training set. This is
  STaR's rationalization step in answer space. Free-text rationale
  bootstrapping was attempted and empirically rejected for this base
  scale: a 124M model fine-tuned to answer format does not reliably emit
  rationales, a finding recorded here deliberately.
- **`rationale`** (for instruction-following bases): the same cycle with
  few-shot rationale generation and STaR filtering, selectable by
  configuration for larger bases.

Every cycle then trains on verified self-completions + hard-mined golds +
replayed gold data, and faces the gate: accept only if held-out accuracy
exceeds the incumbent by more than `accept_margin` (0.004); otherwise
restore the incumbent weights. The evaluation is frozen by constitution —
fixed 250-question validation subset, direct-answer format, greedy decoding
— so accuracy is strictly comparable across every cycle and version.

The genome (`config/genome.json`) holds the mutable hyperparameters
(sampling temperature, attempts per question, batch composition, steps,
learning rate, cooldown). On a fixed cadence the daemon may mutate one
parameter within constitutional bounds; the mutation survives only if the
cycle it acted in passes the gate, otherwise it is reverted. This is the
system's only channel of self-modification.

## Architecture

```
                   operator                          autonomous daemon
        ┌──────────────────────────┐        ┌────────────────────────────────┐
        │  Phase 0 SFT on CSQA     │        │  ┌──────────────────────────┐  │
        │  3,000 real examples     │───────►│  │ 1. SAMPLE next questions │  │
        │  720 steps, eval 36.4%   │  v0    │  │ 2. GENERATE k attempts   │  │
        └──────────────────────────┘        │  │ 3. FILTER vs ground truth│  │
                                            │  │    + hard-mine misses    │  │
        constitution (immutable)            │  │ 4. TRAIN (resumable)     │  │
        ┌──────────────────────────┐        │  │ 5. GATE on held-out set  │  │
        │ base model, task, eval   │        │  │    accept | rollback     │  │
        │ protocol, bounds, margin │◄───────│  │ 6. PUBLISH ledger+git   │  │
        └──────────────────────────┘        │  └──────────┬───────────────┘  │
                                            │             │ accepted          │
        genome (self-modifiable)            │             ▼                   │
        ┌──────────────────────────┐        │   ouroboros-vN checkpoints     │
        │ temperature, k, steps,   │◄───────│   EVOLUTION.md, ledger,        │
        │ lr, replay, cooldown     │ mutate │   curated samples, pushes      │
        └──────────────────────────┘        └────────────────────────────────┘
```

## Results

Held-out accuracy on the fixed 250-question CommonsenseQA validation
subset (greedy decoding; chance is 0.200):

| model | held-out acc | note |
|---|---|---|
| GPT-2 124M, zero-shot | 0.024 | does not follow the answer format; raw continuations are mostly unparseable |
| **ouroboros-v0** (Phase 0 SFT) | **0.364** | operator fine-tune: 3,000 real CSQA train examples, 720 steps |
| ouroboros-vN | see `EVOLUTION.md` | each accepted cycle, autonomously gated |

The live record — every attempt, accepted or rolled back, with accuracies,
training logs, and genome state — is in `EVOLUTION.md` (human-readable) and
`workspace/evals/ledger.jsonl` (machine-readable). Rollbacks are part of
the record by design: the system's honesty is auditable.

Curated data samples from each cycle are committed under
`workspace/curated/sample_cycle_N.jsonl` (attribution below).

## Repository layout

```
ouroboros/            the system (single package, no server, no UI)
  config.py           paths, atomic state, ledger, constitution access
  data.py             CommonsenseQA access, prompt formats, parsing, exemplars
  model.py            SFT trainer (completion-only loss, interruptible), sampling
  phase0.py           operator-run initial fine-tuning (idempotent, resumable)
  cycle.py            the improvement cycle: sample/filter/train/gate/publish
  genome.py           bounded self-modification of training hyperparameters
  evaluate.py         frozen held-out evaluation
  gitops.py           autonomous commits; secret scan; offline-tolerant push
  daemon.py           long-running loop; crash-safe
  cli.py              phase0 | cycle | daemon | eval | status
config/
  genome.json         mutable hyperparameters (the "genome")
  constitution.json   immutable invariants + hard bounds
  remote.json         public repo coordinates (no secrets)
scripts/
  bootstrap.sh        install -> phase 0 -> daemon
  install_cron.sh     schedule one cycle every 30 minutes
workspace/            the system's own state (checkpoints, ledgers, logs)
EVOLUTION.md          append-only evolution record, written by the system
```

## Reproduction

```
git clone https://github.com/Toluwer/ouroboros-rsi
cd ouroboros-rsi
GITHUB_TOKEN=... ./scripts/bootstrap.sh      # deps -> phase 0 -> daemon
```

Single-cycle or scheduled operation:

```
python -m ouroboros cycle                    # one improvement cycle
./scripts/install_cron.sh                    # one cycle every 30 min
python -m ouroboros status                   # state + recent ledger
python -m ouroboros eval                     # score current best checkpoint
```

Offline behavior: the dataset and base model are cached under `.hf_cache/`
after first fetch; training and evaluation need no network. Pushes fail
soft when offline — commits accumulate locally and ship on the next
connected cycle. `GITHUB_TOKEN` is read from the environment at push time
only and is never written to disk; `ouroboros/gitops.py` scans every commit
for credential patterns and aborts on any hit.

Scaling: the base model is a constitution entry, not code. Running with a
larger base (e.g. Qwen2.5-0.5B/1.5B/7B, all Apache-2.0) and
`selfgen_mode: "rationale"` activates full STaR rationale bootstrapping on
the identical cycle code path. Everything except `constitution.json` stays
the same; the loop, gate, genome, and publication machinery are
scale-agnostic.

## Safety

See `SAFETY.md`. In brief: the system modifies itself through exactly one
channel (the genome, bounded, one experiment at a time, empirically
settled); it cannot modify its base model, task, datasets, evaluation
protocol, acceptance rule, or its own code; it executes no generated code;
weights roll back on any failed gate; the operator baseline is never
pruned; and every mutation and cycle is journaled.

## Attribution and license

Code: MIT (see `LICENSE`). Base model GPT-2 (OpenAI, MIT) via
`openai-community/gpt2`. Benchmark: CommonsenseQA (Talmor, Herzig, Murphy
& Berant, NAACL 2019) via `tau/commonsense_qa` on the Hugging Face Hub,
consumed under its distribution terms; committed samples are
model-generated derivatives with attribution, kept small deliberately.
Method lineage: STaR (Zelikman et al. 2022); Noisy Student self-training
(Xie et al. 2020); self-instruct-style bootstrapping (Wang et al. 2022) —
all ground-truth-verified here, which is what the gate enforces.
