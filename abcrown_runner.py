"""Shared helper: run alpha,beta-CROWN in batch mode and parse per-instance results.

Used by run_experiments.py (headline 2-phase comparison) and analyze.py (deep analyses).
One VNNLIB is written per instance and listed in instances.csv (the path the config's
csv_name points at); the verifier loads the model once and verifies all specs.
"""
import os
import re
import subprocess
import sys

from gen_vnnlib import write_vnnlib

REPO = os.path.dirname(os.path.abspath(__file__))
ABCROWN = os.path.join(REPO, "alpha-beta-CROWN", "complete_verifier", "abcrown.py")
CONFIG = os.path.join(REPO, "mnist_fc.yaml")
VNNDIR = os.path.join(REPO, "vnnlib")


def verdict_of(tag):
    """Map an alpha,beta-CROWN result tag to the assignment's three categories."""
    if tag.startswith("safe"):
        return "verified"
    if tag.startswith("unsafe"):
        return "falsified"
    return "timeout"  # unknown / timeout


def run_batch(instances, timeout, extra_args=None):
    """instances: list of (key, x, label, eps). Returns {key: (verdict, tag, seconds)}.

    The raw `tag` (e.g. safe-incomplete, safe, unsafe-pgd, unsafe-bab, unknown) is kept
    so callers can tell apart cheap CROWN bounds vs full branch-and-bound, and PGD vs
    BaB counterexamples.
    """
    os.makedirs(VNNDIR, exist_ok=True)
    keys = []
    with open(os.path.join(REPO, "instances.csv"), "w") as csv_f:
        for key, x, label, eps in instances:
            vp = os.path.join(VNNDIR, f"{key}.vnnlib")
            write_vnnlib(vp, x, label, eps)
            csv_f.write(os.path.relpath(vp, REPO) + "\n")
            keys.append(key)

    cmd = [sys.executable, "-u", ABCROWN, "--config", CONFIG, "--timeout", str(timeout)]
    if extra_args:
        cmd += extra_args
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    found = re.findall(r"Result:\s+(\S+)\s+in\s+([\d.]+)\s+seconds", proc.stdout)
    if len(found) != len(keys):
        sys.stderr.write(proc.stdout[-2000:])
        raise RuntimeError(f"parsed {len(found)} results for {len(keys)} instances")
    return {k: (verdict_of(tag), tag, float(t)) for k, (tag, t) in zip(keys, found)}
