"""Headline two-phase comparison of alpha,beta-CROWN against Marabou (Assignment #3).

  Phase 1 - Sample Scan  : samples 0..99 at eps=0.05      -> which samples are non-robust?
  Phase 2 - Epsilon Sweep: eps 0.01..0.20 on a few samples -> the robustness threshold.

Identical instances to the Marabou study, so the two verifiers can be compared directly.
Output: results/results.md and results/results.csv. Deeper analyses live in analyze.py.
"""
import os

import numpy as np

from abcrown_runner import REPO, run_batch

OUTDIR = os.path.join(REPO, "results")

SCAN_EPSILON = 0.05
SCAN_TIMEOUT = 60
SCAN_SAMPLES = range(100)
SWEEP_EPSILONS = [round(0.01 * k, 2) for k in range(1, 21)]  # 0.01 .. 0.20
SWEEP_TIMEOUT = 300
SWEEP_SAMPLES = [0, 8, 33]  # 0 = robust baseline; 8, 33 = non-robust in Assignment #3


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    X = np.load(os.path.join(REPO, "models", "sample_inputs.npy"))
    Y = np.load(os.path.join(REPO, "models", "sample_labels.npy"))

    print(f"Phase 1: scanning {len(SCAN_SAMPLES)} samples at eps={SCAN_EPSILON}")
    p1 = run_batch(
        [(f"s{i}_e{SCAN_EPSILON}", X[i], int(Y[i]), SCAN_EPSILON) for i in SCAN_SAMPLES],
        SCAN_TIMEOUT,
    )
    falsified = [i for i in SCAN_SAMPLES if p1[f"s{i}_e{SCAN_EPSILON}"][0] == "falsified"]
    print(f"Phase 1 done. non-robust samples: {falsified}")

    print(f"Phase 2: epsilon sweep on samples {SWEEP_SAMPLES}")
    p2 = run_batch(
        [(f"s{s}_e{e}", X[s], int(Y[s]), e) for s in SWEEP_SAMPLES for e in SWEEP_EPSILONS],
        SWEEP_TIMEOUT,
    )

    write_reports(p1, p2, falsified)
    print(f"Results written to {OUTDIR}/")


def write_reports(p1, p2, falsified):
    with open(os.path.join(OUTDIR, "results.csv"), "w") as f:
        f.write("phase,sample,epsilon,verdict,tag,seconds\n")
        for i in SCAN_SAMPLES:
            v, tag, t = p1[f"s{i}_e{SCAN_EPSILON}"]
            f.write(f"1,{i},{SCAN_EPSILON},{v},{tag},{t}\n")
        for s in SWEEP_SAMPLES:
            for e in SWEEP_EPSILONS:
                v, tag, t = p2[f"s{s}_e{e}"]
                f.write(f"2,{s},{e},{v},{tag},{t}\n")

    lines = ["# alpha,beta-CROWN Verification Results\n"]
    lines.append("## Phase 1 - Sample Scan\n")
    lines.append(f"Fixed eps = {SCAN_EPSILON}, timeout = {SCAN_TIMEOUT}s, samples 0-99.\n")
    n_ver = sum(p1[k][0] == "verified" for k in p1)
    n_fal = sum(p1[k][0] == "falsified" for k in p1)
    n_to = sum(p1[k][0] == "timeout" for k in p1)
    mean_t = sum(p1[k][2] for k in p1) / len(p1)
    lines.append(f"- verified: {n_ver}, falsified: {n_fal}, timeout: {n_to}")
    lines.append(f"- non-robust samples: {falsified}")
    lines.append(f"- mean time: {mean_t:.3f}s\n")

    lines.append("## Phase 2 - Epsilon Sweep\n")
    lines.append("Verdict per (sample, epsilon); 'threshold' = smallest eps that is falsified.\n")
    lines.append("| sample | " + " | ".join(f"{e}" for e in SWEEP_EPSILONS) + " | threshold |")
    lines.append("|" + "---|" * (len(SWEEP_EPSILONS) + 2))
    for s in SWEEP_SAMPLES:
        row, threshold = [], "-"
        for e in SWEEP_EPSILONS:
            v = p2[f"s{s}_e{e}"][0]
            row.append({"verified": "V", "falsified": "F", "timeout": "T"}[v])
            if threshold == "-" and v == "falsified":
                threshold = str(e)
        lines.append(f"| {s} | " + " | ".join(row) + f" | {threshold} |")
    lines.append("\nV = verified (robust), F = falsified (counterexample), T = timeout.\n")

    with open(os.path.join(OUTDIR, "results.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
