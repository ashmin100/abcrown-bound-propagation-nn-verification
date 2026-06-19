"""Minimal demo: run alpha,beta-CROWN on the mnist_fc model.

Verifies two MNIST samples under an L-inf ball of radius eps: one robust case
(expected 'verified') and one non-robust case (expected 'falsified'). This is just a
smoke test of the setup; the full study lives in run_experiments.py and analyze.py.

Usage: python test.py
"""
import os

import numpy as np

from abcrown_runner import REPO, run_batch

EPS = 0.05
SAMPLES = (0, 8)  # 0 is robust at eps=0.05; 8 is not (Assignment #3)

X = np.load(os.path.join(REPO, "models", "sample_inputs.npy"))
Y = np.load(os.path.join(REPO, "models", "sample_labels.npy"))

res = run_batch([(f"s{i}_e{EPS}", X[i], int(Y[i]), EPS) for i in SAMPLES], timeout=60)

print(f"\nalpha,beta-CROWN on mnist_fc  (L-inf, eps={EPS})")
print(f"{'sample':>6} {'label':>5} {'verdict':>10} {'tag':>16} {'time(s)':>8}")
for i in SAMPLES:
    verdict, tag, t = res[f"s{i}_e{EPS}"]
    print(f"{i:>6} {int(Y[i]):>5} {verdict:>10} {tag:>16} {t:>8.3f}")
