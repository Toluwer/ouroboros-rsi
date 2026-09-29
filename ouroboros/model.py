"""Model wrapper: the three primitives Ouroboros needs.

  * supervised fine-tuning with loss restricted to completion tokens
  * sampling (k completions per prompt)
  * checkpoint save / load

The trainer is a deliberately small, explicit PyTorch loop rather than
Transformers' Trainer: every step is inspectable, there is no hidden
behavior, and the same code path runs on CPU or GPU. Gradient checkpointing
is enabled by default so the 124M-parameter default model fine-tunes
comfortably within 4 GB of RAM.
"""

from __future__ import annotations

import gc
import math
import os
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(max(1, os.cpu_count() or 2))

MAX_SEQ_LEN = 256
GRADIENT_CHECKPOINTING = True  # RAM safety; negligible effect at this scale


class OuroborosModel:
    def __init__(self, model_name: str = "openai-community/gpt2"):
        self.name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if GRADIENT_CHECKPOINTING:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False
        self.model.config.pad_token_id = self.tok.pad_token_id
        self.model.generation_config.pad_token_id = self.tok.pad_token_id

    # ------------------------------------------------------------------
    # Supervised fine-tuning
    # ------------------------------------------------------------------
    def fine_tune(
        self,
        examples: list[tuple[str, str]],
        steps: int,
        lr: float,
        batch_size: int = 8,
        seed: int = 0,
        log=print,
        max_seconds: float | None = None,
    ) -> dict:
        """Fine-tune on (prompt, completion) pairs; loss on completion only.

        Linear warmup over the first 10% of steps, cosine decay to 0,
        gradient clipping at 1.0, AdamW with weight decay 0.01.

        Training is interruptible: if ``max_seconds`` is given, the loop
        stops cleanly at the next step boundary once the budget is spent,
        and the returned dict reports the number of steps actually run so
        the caller can checkpoint and resume. Chunked continuation
        reinitializes the optimizer (AdamW state is not persisted), which
        is equivalent to a warm restart.
        """
        rng = random.Random(seed)
        torch.manual_seed(seed)

        tokenized: list[tuple[list[int], list[int]]] = []
        for prompt, completion in examples:
            p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
            c_ids = self.tok(completion, add_special_tokens=False)["input_ids"] + [
                self.tok.eos_token_id
            ]
            ids = (p_ids + c_ids)[:MAX_SEQ_LEN]
            labels = ([-100] * len(p_ids) + c_ids)[:MAX_SEQ_LEN]
            tokenized.append((ids, labels))

        self.model.train()
        self.model.config.use_cache = False
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        warmup = max(1, steps // 10)

        def lr_lambda(s: int) -> float:
            if s < warmup:
                return s / warmup
            prog = (s - warmup) / max(1, steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * prog))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        losses: list[float] = []
        step = 0
        batch: list[tuple[list[int], list[int]]] = []
        t_start = time.time()

        def run_batch() -> float:
            maxlen = max(len(ids) for ids, _ in batch)
            pad = self.tok.pad_token_id
            input_ids, attn, labels = [], [], []
            for ids, lab in batch:
                padlen = maxlen - len(ids)
                input_ids.append(ids + [pad] * padlen)
                attn.append([1] * len(ids) + [0] * padlen)
                labels.append(lab + [-100] * padlen)
            out = self.model(
                input_ids=torch.tensor(input_ids, dtype=torch.long),
                attention_mask=torch.tensor(attn, dtype=torch.long),
                labels=torch.tensor(labels, dtype=torch.long),
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            return out.loss.item()

        while step < steps:
            order = list(range(len(tokenized)))
            rng.shuffle(order)
            for i in order:
                batch.append(tokenized[i])
                if len(batch) == batch_size:
                    losses.append(run_batch())
                    batch = []
                    step += 1
                    if step % 25 == 0 or step == steps:
                        recent = losses[-25:]
                        log(
                            f"    step {step}/{steps} "
                            f"loss {sum(recent) / len(recent):.4f}"
                        )
                    if step >= steps:
                        break
                    if (
                        max_seconds is not None
                        and (time.time() - t_start) > max_seconds
                    ):
                        break
            if (
                max_seconds is not None
                and (time.time() - t_start) > max_seconds
            ):
                break
            if not tokenized:
                break
            # flush a partial batch (fewer examples than batch_size)
            if batch and step < steps:
                losses.append(run_batch())
                batch = []
                step += 1
                if step >= steps:
                    break

        del opt, sched
        gc.collect()
        self.model.eval()
        self.model.config.use_cache = not GRADIENT_CHECKPOINTING
        recent = losses[-25:]
        return {
            "steps": step,
            "requested_steps": steps,
            "lr": lr,
            "batch_size": batch_size,
            "final_loss": sum(recent) / len(recent) if recent else None,
            "examples": len(examples),
            "interrupted": step < steps,
        }

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        k: int = 1,
        temperature: float = 0.7,
        max_new_tokens: int = 64,
        do_sample: bool = True,
    ) -> list[str]:
        self.model.eval()
        enc = self.tok(prompt, return_tensors="pt")
        kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tok.pad_token_id,
            eos_token_id=self.tok.eos_token_id,
        )
        if do_sample:
            kwargs.update(do_sample=True, temperature=max(1e-3, temperature), top_p=0.95,
                          num_return_sequences=k)
        else:
            kwargs.update(do_sample=False, num_return_sequences=1)
        with torch.no_grad():
            out = self.model.generate(**enc, **kwargs)
        plen = enc.input_ids.shape[1]
        return [self.tok.decode(o[plen:], skip_special_tokens=True) for o in out]

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save(self, path) -> None:
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)

    @classmethod
    def load(cls, path: str) -> "OuroborosModel":
        return cls(model_name=str(path))
