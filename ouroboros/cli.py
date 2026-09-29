"""Command-line interface.

    python -m ouroboros phase0 [--fast]   operator-run initial fine-tune
    python -m ouroboros cycle  [--fast]   run one improvement cycle
    python -m ouroboros daemon [--fast]   run improvement cycles forever
    python -m ouroboros eval   [--n N]    score the current best checkpoint
    python -m ouroboros status            state + recent ledger entries
"""

from __future__ import annotations

import argparse
import json

from . import config, data, evaluate


def main() -> None:
    ap = argparse.ArgumentParser(prog="ouroboros", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("phase0", help="operator-run initial fine-tune")
    p0.add_argument("--fast", action="store_true", help="debug-sized profile")
    p0.add_argument("--budget", type=float, default=None,
                    help="time budget in seconds; training checkpoints and resumes")

    pc = sub.add_parser("cycle", help="run one improvement cycle (resumable)")
    pc.add_argument("--fast", action="store_true")
    pc.add_argument("--budget", type=float, default=None,
                    help="time budget in seconds; the cycle resumes on next call")

    pd = sub.add_parser("daemon", help="run improvement cycles forever")
    pd.add_argument("--fast", action="store_true")

    pe = sub.add_parser("eval", help="score the current best checkpoint")
    pe.add_argument("--n", type=int, default=None)

    sub.add_parser("status", help="print state and recent ledger entries")

    args = ap.parse_args()

    if args.cmd == "phase0":
        from .phase0 import run_phase0

        run_phase0(fast=args.fast, budget_s=args.budget)
    elif args.cmd == "cycle":
        from .cycle import run_cycle

        run_cycle(fast=args.fast, budget_s=args.budget)
    elif args.cmd == "daemon":
        from .daemon import run

        run(fast=args.fast)
    elif args.cmd == "eval":
        from .model import OuroborosModel

        state = config.load_state()
        constitution = config.load_constitution()
        imm = constitution["immutable"]
        n = args.n or imm["eval_n"]
        model = OuroborosModel.load(state["best_ckpt"])
        ev = evaluate.evaluate(model, data.eval_subset(n, imm["eval_seed"]))
        ev.pop("details", None)
        print(json.dumps(ev, indent=2))
    elif args.cmd == "status":
        from .daemon import status

        print(json.dumps(status(), indent=2))


if __name__ == "__main__":
    main()
