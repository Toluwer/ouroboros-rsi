"""Dataset access for Ouroboros.

Benchmark: CommonsenseQA (Talmor et al., NAACL 2019) -- a real, widely used
5-way multiple-choice benchmark with 9,741 train and 1,221 validation
questions sourced from ConceptNet. It is fetched once from the Hugging Face
Hub (dataset id ``tau/commonsense_qa``) and cached locally under .hf_cache,
after which every cycle runs fully offline.

Two prompt formats are used:

  direct      "Question: ...\nA. ...\n...\nE. ...\nAnswer:"
              Completion: " ignore (A)" -- the answer text followed by its
              letter. The answer word carries the semantic gradient (the
              format the original CSQA paper used for GPT fine-tuning);
              the parenthesized letter makes predictions cheap to parse.
              Used for phase-0 SFT, replay, and all held-out evaluation
              (so accuracy is comparable across cycles).

  rationale   "Question: ...\nA. ...\n...\nE. ...\nReasoning:"
              Completion: " {rationale}\nAnswer: ignore (A)". Used by the
              self-improvement loop; STaR-style rationales are kept only
              when they lead to the ground-truth answer, or are produced by
              rationalization (gold answer shown as a hint).

Ground-truth labels are never shown to the model at generation time; they
are used only to filter completions afterwards, exactly as in STaR
(Zelikman et al., 2022).
"""

from __future__ import annotations

import json
import random

from . import config

CHOICE_LETTERS = ["A", "B", "C", "D", "E"]
DATASET_ID = "tau/commonsense_qa"


def _to_item(rec: dict) -> dict:
    labels = list(rec["choices"]["label"])
    texts = list(rec["choices"]["text"])
    by_label = dict(zip(labels, texts))
    choices = [(l, by_label[l]) for l in CHOICE_LETTERS if l in by_label]
    return {
        "question": rec["question"].strip(),
        "choices": choices,
        "answer": rec["answerKey"].strip(),
    }


def _load_split(split: str) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split=split)
    return [_to_item(r) for r in ds]


def train_items() -> list[dict]:
    return _load_split("train")


def validation_items() -> list[dict]:
    return _load_split("validation")


def eval_subset(n: int, seed: int) -> list[dict]:
    """Fixed, deterministic held-out subset, cached to disk on first use so
    every cycle is scored against exactly the same questions."""
    path = config.W_EVALS / f"eval_subset_{n}_{seed}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    items = validation_items()
    rng = random.Random(seed)
    subset = rng.sample(items, min(n, len(items)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=1)
    return subset


def format_choices(item: dict) -> str:
    return "\n".join(f"{letter}. {text}" for letter, text in item["choices"])


def direct_prompt(item: dict) -> str:
    return (
        f"Question: {item['question']}\n"
        f"{format_choices(item)}\n"
        "Answer:"
    )


def rationale_prompt(item: dict) -> str:
    """Zero-shot rationale prompt. Used for TRAINING pairs (the model
    internalizes the pattern from curated data); generation-time prompts
    prepend few-shot exemplars (see few_shot_prefix)."""
    return (
        f"Question: {item['question']}\n"
        f"{format_choices(item)}\n"
        "Reasoning:"
    )


# Two hand-written rationales over real CommonsenseQA train questions, used
# only as few-shot exemplars at generation time (STaR-style bootstrapping;
# the model's own filtered outputs, not these exemplars, are the training
# data). Questions: CSQA train indices 0 and 5.
EXEMPLARS = [
    {
        "question": "The sanctions against the school were a punishing blow, "
        "and they seemed to what the efforts the school had made to change?",
        "choices": [("A", "ignore"), ("B", "enforce"), ("C", "authoritarian"),
                    ("D", "yell at"), ("E", "avoid")],
        "answer": "A",
        "reasoning": "Sanctions are a punishment meant to pressure a school "
        "into changing. If the school had already made efforts to change, "
        "the sanctions would work against those efforts instead of "
        "supporting them, so they seemed to disregard the efforts. "
        "The word that fits is ignore.",
    },
    {
        "question": "What home entertainment equipment requires cable?",
        "choices": [("A", "radio shack"), ("B", "substation"), ("C", "cabinet"),
                    ("D", "television"), ("E", "desk")],
        "answer": "D",
        "reasoning": "Home entertainment equipment is something found in a "
        "living room used for watching shows. Cable service delivers "
        "channels to a screen, and the device that needs a cable "
        "connection to receive them is a television. The other options "
        "are furniture or utility structures.",
    },
]


def few_shot_prefix() -> str:
    parts = []
    for ex in EXEMPLARS:
        parts.append(
            f"Question: {ex['question']}\n"
            + "\n".join(f"{l}. {t}" for l, t in ex["choices"])
            + f"\nReasoning: {ex['reasoning']}\n"
            + f"Answer: {answer_text(ex)} ({ex['answer']})\n"
        )
    return "\n".join(parts) + "\n"


def fs_rationale_prompt(item: dict) -> str:
    """Few-shot generation prompt for sampling rationales."""
    return few_shot_prefix() + rationale_prompt(item)


def fs_rationalization_prompt(item: dict, gold: str) -> str:
    """Few-shot STaR rationalization: the gold answer is given as a hint and
    the model must produce a supporting rationale for it."""
    return (
        few_shot_prefix()
        + f"Question: {item['question']}\n"
        + format_choices(item)
        + f"\nAnswer: {answer_text_for(item, gold)} ({gold})\n"
        "Reasoning:"
    )


def answer_text_for(item: dict, letter: str) -> str:
    for l, t in item["choices"]:
        if l == letter:
            return t.strip()
    return ""


def rationale_completion(item: dict, rationale: str) -> str:
    return f" {rationale.strip()}\nAnswer: {answer_text(item)} ({item['answer']})"


def direct_completion(item: dict) -> str:
    return f" {answer_text(item)} ({item['answer']})"


def answer_text(item: dict) -> str:
    for letter, text in item["choices"]:
        if letter == item["answer"]:
            return text.strip()
    return ""


def parse_letter(text: str) -> str | None:
    """Extract a predicted choice letter from generated text.

    Order of preference: explicit parenthesized letter ("... (B)"), the
    letter after an "Answer:" marker, a standalone letter, then a single
    leading letter. Used for filtering self-generated rationales.
    """
    if not text:
        return None
    import re

    m = re.search(r"\(([A-E])\)", text)
    if m:
        return m.group(1)
    m = re.search(r"Answer:\s*(?:[\w\s]{0,30}?)\(?([A-E])\)?(?![a-zA-Z])", text)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-E])\b", text)
    if m:
        return m.group(1)
    first = text.strip()[:1]
    return first if first in "ABCDE" else None


def parse_prediction(text: str, item: dict) -> str | None:
    """Eval-side prediction: parenthesized letter first, then choice-text
    containment, then the generic letter parse."""
    if not text:
        return None
    import re

    m = re.search(r"\(([A-E])\)", text)
    if m:
        return m.group(1)
    t = text.lower()
    for letter, choice in item["choices"]:
        c = choice.strip().lower()
        if c and c in t:
            return letter
    return parse_letter(text)
