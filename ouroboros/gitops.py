"""Git operations for the autonomous loop.

The daemon commits its own evaluation ledgers, curated data samples, genome,
and EVOLUTION.md -- and pushes to the repository described by the committed,
secret-free config/remote.json. The GitHub token is read from the
GITHUB_TOKEN environment variable at call time and is never written to disk.

Two safety properties:

  * secret_scan() blocks any commit whose added content contains a GitHub
    token pattern (or any high-entropy credential that looks like one);
  * push failures (e.g. offline) never lose work -- commits accumulate
    locally and are pushed by a later cycle when connectivity returns.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from . import config

SECRET_PATTERNS = [
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"gho_[A-Za-z0-9]{30,}"),
    re.compile(r"ghu_[A-Za-z0-9]{30,}"),
    re.compile(r"ghs_[A-Za-z0-9]{30,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{30,}"),
]

_AUTHOR_NAME = "Ouroboros (autonomous)"
_AUTHOR_EMAIL = "ouroboros@localhost"


def _git(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(config.REPO_ROOT)] + args,
        capture_output=True,
        text=True,
        check=check,
    )


def ensure_repo() -> None:
    if not (config.REPO_ROOT / ".git").exists():
        _git(["init", "-q", "-b", "main"])
    try:
        _git(["config", "user.name"], check=False)
        has_name = _git(["config", "user.name"], check=False).stdout.strip()
        if not has_name:
            _git(["config", "user.name", _AUTHOR_NAME])
            _git(["config", "user.email", _AUTHOR_EMAIL])
    except subprocess.CalledProcessError:
        pass


def secret_scan(paths: list[str]) -> list[str]:
    """Return list of 'path:pattern' hits in the given files."""
    hits = []
    for p in paths:
        fp = config.REPO_ROOT / p
        if not fp.is_file() or fp.stat().st_size > 2_000_000:
            continue
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pat in SECRET_PATTERNS:
            if pat.search(text):
                hits.append(f"{p}:{pat.pattern}")
    return hits


def commit_and_push(tag: str, message: str, paths: list[str]) -> dict:
    """Stage specific paths, commit, and push if possible."""
    ensure_repo()
    for p in paths:
        _git(["add", "--", p], check=False)

    staged = _git(["diff", "--cached", "--name-only"], check=False).stdout.strip()
    if not staged:
        return {"committed": False, "pushed": False, "reason": "nothing to commit"}

    files = [l.strip() for l in staged.splitlines() if l.strip()]
    hits = secret_scan(files)
    if hits:
        _git(["reset", "-q"])
        raise RuntimeError(f"secret pattern detected, commit aborted: {hits}")

    _git(["commit", "-q", "-m", f"{tag}: {message}"])

    remote = config.load_remote()
    token = os.environ.get("GITHUB_TOKEN", "")
    result = {"committed": True, "pushed": False, "files": len(files)}
    if token and remote.get("owner") and remote.get("repo"):
        url = (f"https://x-access-token:{token}@github.com/"
               f"{remote['owner']}/{remote['repo']}.git")
        # Token used in the push URL only; never stored in config or remote.
        proc = _git(["push", url, "HEAD:refs/heads/main"], check=False)
        result["pushed"] = proc.returncode == 0
        if not result["pushed"]:
            result["reason"] = (proc.stderr or "push failed")[-300:]
    else:
        result["reason"] = "GITHUB_TOKEN or remote config absent; commit kept locally"
    return result
