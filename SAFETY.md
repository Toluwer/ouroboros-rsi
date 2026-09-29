# Safety Model

Ouroboros modifies its own training configuration and produces its own
training data. That capability demands an explicit, inspectable safety
envelope. This document defines it and how it is enforced.

## Bounded self-modification

The system may only change itself through one channel: the genome
(`config/genome.json`), which holds sampling and training hyperparameters.
Every mutation is constrained four ways:

1. **Hard bounds.** `config/constitution.json` declares minimum and maximum
   values for every mutable parameter. Mutations are clamped to these
   bounds by code, not by convention.
2. **One experiment at a time, on a fixed cadence** (one per
   `mutation_every` cycles). The system cannot stack simultaneous changes.
3. **Empirical settlement.** A mutation survives only if the cycle it
   acted in passes the held-out acceptance gate. A rejected cycle reverts
   both the weights and the mutation.
4. **Full audit trail.** Every mutation is recorded in the cycle ledger
   and in EVOLUTION.md, with old value, new value, and outcome.

## Constitutional invariants

The daemon can never modify, at runtime: the base model, the task, the
training and evaluation datasets, the evaluation protocol (fixed subset,
direct-answer format, greedy decoding), the acceptance margin, or the
bounds themselves. Changing any of these requires a human commit to this
repository. The constitution file is the boundary of the system's
authority over itself.

## No arbitrary code execution

Self-improvement here means updating weights and hyperparameters through
the fixed, reviewed code path in `ouroboros/`. The daemon does not write
or execute new code, does not shell out beyond `git` (for its own
artifacts), and does not modify its own source. This is a deliberate
restriction: it keeps the improvement loop auditable end to end while
still giving the system genuine authority over its learning process.

## Credential hygiene

The GitHub token is read from the `GITHUB_TOKEN` environment variable at
push time only. It is never written to disk, never committed, and
`ouroboros/gitops.py` runs a secret-pattern scan over every commit's
staged content and aborts the commit if any credential pattern is found.
The remote described in `config/remote.json` contains only public
information.

## Data provenance

All training data derives from two sources: the CommonsenseQA training
split (real, published benchmark, consumed via the Hugging Face Hub) and
the model's own filtered generations. Committed samples carry attribution
in the README. No synthetic or hand-rolled substitute data is used
anywhere in the pipeline.

## Rollback

Weights are versioned per accepted cycle (`workspace/checkpoints/ouroboros-vN`).
A rejected cycle restores the incumbent checkpoint immediately, so the
system's best weights are never degraded by an exploratory failure, and
the operator baseline `ouroboros-v0` is never pruned.
