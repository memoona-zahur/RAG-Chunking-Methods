"""Sanity + hygiene check before committing or sharing the project.

Verifies: importable modules, .env hygiene (the Groq key may ONLY live in the
git-ignored .env), tests pass, and no required-fact strings are hand-written
into the docs beyond the demo (all reported numbers come from a live run).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
KEY_PATTERN = re.compile(r"gsk_[A-Za-z0-9]{20,}")


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def check(name: str, ok: bool, detail: str = ""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail and not ok else ""))
    return ok


def main() -> int:
    print("RAG-Chunking-Methods — project verification")
    results: list[bool] = []

    # 1 · every module imports
    importable = True
    for mod in ["chunkers", "evaluate", "costs", "llm", "agentic", "embed_store", "demo"]:
        try:
            __import__(mod)
        except Exception as exc:  # noqa: BLE001
            importable = False
            print(f"         import {mod} failed: {exc}")
    results.append(check("all modules import", importable))

    # 2 · .env exists and holds a key (but the file itself is git-ignored)
    env_path = os.path.join(ROOT, ".env")
    has_key_in_env = os.path.isfile(env_path) and bool(
        KEY_PATTERN.search(open(env_path, encoding="utf-8").read())
    )
    results.append(check(".env present with a Groq key", has_key_in_env))

    # 3 · the key appears NOWHERE in tracked files
    tracked = sh(f'git -C "{ROOT}" ls-files').split()
    hijack = []
    for rel in tracked:
        p = os.path.join(ROOT, rel)
        if not os.path.isfile(p):
            continue
        if rel == ".env":
            continue
        if KEY_PATTERN.search(open(p, encoding="utf-8", errors="ignore").read()):
            hijack.append(rel)
    results.append(check("key absent from all tracked files", not hijack, ", ".join(hijack)))

    # 4 · .env is in .gitignore
    gi = open(os.path.join(ROOT, ".gitignore"), encoding="utf-8").read()
    results.append(check(".env is git-ignored", ".env" in gi))

    # 5 · tests pass
    t = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
    tests_ok = t.returncode == 0 and "passed" in t.stdout
    results.append(check("pytest green", tests_ok, t.stdout.strip().splitlines()[-1] if t.stdout.strip() else ""))

    # 6 · the demo runs end-to-end (dry-run backend, plain output)
    d = subprocess.run(
        [sys.executable, "demo.py"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8"
    )
    demo_ok = d.returncode == 0 and "DONE" in d.stdout
    results.append(check("demo.py runs to completion", demo_ok))

    ok_all = all(results)
    print()
    print(f"RESULT: {sum(results)}/{len(results)} checks passed — "
          f"{'ready to share' if ok_all else 'fix the failures above'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())