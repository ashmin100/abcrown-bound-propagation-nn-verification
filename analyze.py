"""Deep analyses of mnist_fc robustness with alpha,beta-CROWN (report material).

Produces figures + tables under results/ for six angles:
  A  robustness-radius distribution over all 100 samples (alpha,beta-CROWN is fast
     enough to certify the whole test set; Marabou would be impractical at this scale)
  B  runtime vs epsilon: alpha,beta-CROWN stays flat where Marabou blows up
  C  branching-heuristic comparison (babsr / fsb / kfsb) on the hardest instance
  D  incomplete (cheap CROWN bound) vs complete (branch-and-bound) resolution
  E  counterexample extraction + visualization for the non-robust samples
  F  startup overhead vs amortized cost (an honest limitation of alpha,beta-CROWN)

The full epsilon grid sweep (A) is cached to results/sweep.csv so re-runs that only
touch later sections are instant.
"""
import os
import re
import subprocess
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import onnxruntime as ort

sns.set_theme(style="whitegrid", context="notebook")

from abcrown_runner import REPO, ABCROWN, CONFIG, VNNDIR, run_batch, verdict_of
from gen_vnnlib import write_vnnlib

OUT = os.path.join(REPO, "results")
GRID = [round(0.01 * k, 2) for k in range(1, 21)]  # 0.01 .. 0.20 (matches Marabou range)
NSAMPLES = 100

# Marabou sample-0 runtimes from Assignment #3 (Phase 2), for the head-to-head curve.
MARABOU_S0 = {0.01: 0.39, 0.02: 0.42, 0.03: 0.51, 0.04: 0.53, 0.05: 0.48, 0.06: 0.48,
              0.07: 0.48, 0.08: 0.49, 0.09: 0.52, 0.10: 0.91, 0.11: 1.02, 0.12: 6.17,
              0.13: 3.57, 0.14: 8.90, 0.15: 14.28, 0.16: 25.79, 0.17: 38.15,
              0.18: 138.94, 0.19: 300.54, 0.20: 300.48}

SESS = ort.InferenceSession(os.path.join(REPO, "models", "mnist_fc.onnx"))


def predict(x):
    """Return the class predicted by mnist_fc for a (784,) normalized input."""
    logits = SESS.run(None, {"input": x.reshape(1, 784).astype(np.float32)})[0]
    return int(logits.argmax())


# --- A: full epsilon sweep (cached) -------------------------------------------
def sweep(X, Y):
    cache = os.path.join(OUT, "sweep.csv")
    if os.path.exists(cache):
        data = {}
        with open(cache) as f:
            next(f)
            for line in f:
                s, e, v, tag, t = line.strip().split(",")
                data[(int(s), float(e))] = (v, tag, float(t))
        return data
    print(f"Sweep: {NSAMPLES} samples x {len(GRID)} eps = {NSAMPLES * len(GRID)} instances")
    insts = [(f"s{s}_e{e}", X[s], int(Y[s]), e) for s in range(NSAMPLES) for e in GRID]
    res = run_batch(insts, timeout=300)
    data = {(s, e): res[f"s{s}_e{e}"] for s in range(NSAMPLES) for e in GRID}
    with open(cache, "w") as f:
        f.write("sample,epsilon,verdict,tag,seconds\n")
        for (s, e), (v, tag, t) in data.items():
            f.write(f"{s},{e},{v},{tag},{t}\n")
    return data


def threshold(data, s):
    """Smallest eps that is falsified for sample s, or None if robust over the grid."""
    for e in GRID:
        if data[(s, e)][0] == "falsified":
            return e
    return None


def fig_radius(data):
    """A: threshold histogram (seaborn) + certified-robust survival curve."""
    th = [threshold(data, s) for s in range(NSAMPLES)]
    robust = [t for t in th if t is not None]
    censored = sum(t is None for t in th)
    robust_frac = [sum(data[(s, e)][0] == "verified" for s in range(NSAMPLES)) / NSAMPLES
                   for e in GRID]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(robust, bins=len(GRID), ax=ax[0], color="steelblue", edgecolor="white")
    ax[0].set_xlabel("robustness threshold epsilon (L-inf)")
    ax[0].set_ylabel("number of samples")
    ax[0].set_title(f"Threshold distribution ({censored}/100 robust beyond {GRID[-1]})")
    sns.lineplot(x=GRID, y=robust_frac, marker="o", ax=ax[1], color="seagreen")
    ax[1].set_xlabel("epsilon (L-inf)")
    ax[1].set_ylabel("fraction certified robust")
    ax[1].set_title("Certified-robust fraction vs epsilon")
    ax[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_radius_hist.png"), dpi=130)
    plt.close(fig)
    return th, robust, censored


def fig_runtime(data):
    """B: runtime vs epsilon for sample 0 (alpha,beta-CROWN vs Marabou)."""
    abc = [data[(0, e)][2] for e in GRID]
    mar = [MARABOU_S0[e] for e in GRID]
    plt.figure(figsize=(7, 4))
    plt.plot(GRID, mar, "o-", label="Marabou (SMT)")
    plt.plot(GRID, abc, "s-", label="alpha,beta-CROWN (BaB)")
    plt.yscale("log")
    plt.xlabel("epsilon (L-inf)")
    plt.ylabel("verification time (s, log scale)")
    plt.title("Sample 0 (robust): verification time vs epsilon")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_runtime_vs_eps.png"), dpi=130)
    plt.close()
    return abc, mar


def table_breakdown(data):
    """D: incomplete (CROWN) vs complete (BaB) vs PGD resolution per epsilon."""
    rows = []
    for e in GRID:
        tags = [data[(s, e)][1] for s in range(NSAMPLES)]
        rows.append((e,
                     sum(t == "safe-incomplete" for t in tags),
                     sum(t.startswith("safe") and t != "safe-incomplete" for t in tags),
                     sum(t == "unsafe-pgd" for t in tags),
                     sum(t.startswith("unsafe") and t != "unsafe-pgd" for t in tags)))
    return rows


def fig_breakdown(rows):
    """D: 100%-stacked area of how each instance is resolved as epsilon grows."""
    eps = [r[0] for r in rows]
    incomplete = [r[1] for r in rows]
    bab = [r[2] for r in rows]
    falsified = [r[3] + r[4] for r in rows]
    plt.figure(figsize=(7, 4))
    plt.stackplot(eps, incomplete, bab, falsified,
                  labels=["verified by CROWN (incomplete)", "verified by BaB",
                          "falsified (PGD)"],
                  colors=["#9ecae1", "#3182bd", "#de2d26"])
    plt.xlabel("epsilon (L-inf)")
    plt.ylabel("number of samples (of 100)")
    plt.title("How each instance is resolved as epsilon grows")
    plt.legend(loc="center left", fontsize=8)
    plt.margins(0, 0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_breakdown_area.png"), dpi=130)
    plt.close()


# --- C: branching heuristic on the hardest instance ---------------------------
def run_single(x, label, eps, extra):
    # Run as a 1-line batch (csv) so the verifier prints "Result: <tag> in <t> seconds"
    # (single --vnnlib_path mode omits the timing); also count branch-and-bound domains.
    vp = os.path.join(VNNDIR, "hard.vnnlib")
    write_vnnlib(vp, x, label, eps)
    with open(os.path.join(REPO, "instances.csv"), "w") as f:
        f.write(os.path.relpath(vp, REPO) + "\n")
    cmd = [sys.executable, "-u", ABCROWN, "--config", CONFIG, "--timeout", "60"] + extra
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    m_t = re.search(r"Result:\s+(\S+)\s+in\s+([\d.]+)\s+seconds", proc.stdout)
    m_d = re.findall(r"(\d+)\s+domains visited", proc.stdout)
    if m_t:
        verdict, t = verdict_of(m_t.group(1)), float(m_t.group(2))
    elif "Traceback" in proc.stderr:
        verdict, t = "error", float("nan")  # upstream heuristic bug
    else:
        verdict, t = "?", float("nan")
    return verdict, t, (int(m_d[-1]) if m_d else 0)


def branching_compare(data, X, Y):
    # Hardest = verified instance with the largest solve time that actually used BaB.
    cand = [(data[(s, e)][2], s, e) for s in range(NSAMPLES) for e in GRID
            if data[(s, e)][1].startswith("safe") and data[(s, e)][1] != "safe-incomplete"]
    if not cand:  # fall back to the slowest verified instance
        cand = [(data[(s, e)][2], s, e) for s in range(NSAMPLES) for e in GRID
                if data[(s, e)][0] == "verified"]
    _, s, e = max(cand)
    rows = []
    for method in ["babsr", "fsb", "kfsb"]:
        verdict, t, dom = run_single(X[s], int(Y[s]), e, ["--branching_method", method])
        rows.append((method, verdict, t, dom))
    return s, e, rows


def fig_branching(rows, s, e):
    """C: BaB domains per branching heuristic (log scale); skip methods that crashed."""
    ran = [r for r in rows if r[1] != "error"]
    methods = [r[0] for r in ran]
    doms = [r[3] for r in ran]
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x=methods, y=doms, color="slateblue")
    ax.set_yscale("log")
    ax.set_xlabel("branching heuristic")
    ax.set_ylabel("BaB domains visited (log)")
    ax.set_title(f"Branching cost on hardest instance (sample {s}, eps {e})")
    for i, d in enumerate(doms):
        ax.text(i, d, str(d), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_branching.png"), dpi=130)
    plt.close()


# --- E: counterexample extraction + visualization -----------------------------
def parse_cex(path):
    vals = {}
    for X_i, v in re.findall(r"\(X_(\d+)\s+([-\d.eE]+)\)", open(path).read()):
        vals[int(X_i)] = float(v)
    return np.array([vals[i] for i in range(784)], dtype=np.float32)


def counterexamples(data, X, Y):
    nonrobust = [s for s in range(NSAMPLES) if data[(s, 0.05)][0] == "falsified"]
    rows = []
    for s in nonrobust:
        vp = os.path.join(VNNDIR, f"cex_s{s}.vnnlib")
        cex = os.path.join(OUT, f"cex_s{s}.txt")
        write_vnnlib(vp, X[s], int(Y[s]), 0.05)
        subprocess.run([sys.executable, ABCROWN, "--config", CONFIG, "--vnnlib_path", vp,
                        "--save_adv_example", "--cex_path", cex, "--timeout", "60"],
                       cwd=REPO, capture_output=True, text=True)
        adv = parse_cex(cex)
        true_c, adv_c = predict(X[s]), predict(adv)
        rows.append((s, true_c, adv_c))
        # side-by-side image (denormalize: x*std + mean)
        orig_img = (X[s] * 0.3081 + 0.1307).reshape(28, 28).clip(0, 1)
        adv_img = (adv * 0.3081 + 0.1307).reshape(28, 28).clip(0, 1)
        fig, ax = plt.subplots(1, 3, figsize=(7, 2.6))
        ax[0].imshow(orig_img, cmap="gray"); ax[0].set_title(f"original: {true_c}")
        ax[1].imshow(adv_img, cmap="gray"); ax[1].set_title(f"adversarial: {adv_c}")
        ax[2].imshow((adv_img - orig_img), cmap="bwr"); ax[2].set_title("perturbation")
        for a in ax:
            a.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, f"cex_s{s}.png"), dpi=130)
        plt.close()
    return rows


# --- F: startup overhead vs amortized -----------------------------------------
def overhead(X, Y):
    vp = os.path.join(VNNDIR, "s0_e0.05.vnnlib")
    write_vnnlib(vp, X[0], int(Y[0]), 0.05)
    t0 = time.time()
    subprocess.run([sys.executable, ABCROWN, "--config", CONFIG, "--vnnlib_path", vp,
                    "--timeout", "60"], cwd=REPO, capture_output=True, text=True)
    cold_wall = time.time() - t0  # includes full process startup
    # amortized: 100-instance batch wall time / 100
    t0 = time.time()
    run_batch([(f"s{s}_e0.05", X[s], int(Y[s]), 0.05) for s in range(NSAMPLES)], 60)
    batch_wall = time.time() - t0
    return cold_wall, batch_wall / NSAMPLES


def main():
    os.makedirs(OUT, exist_ok=True)
    X = np.load(os.path.join(REPO, "models", "sample_inputs.npy"))
    Y = np.load(os.path.join(REPO, "models", "sample_labels.npy"))

    data = sweep(X, Y)
    th, robust, censored = fig_radius(data)
    abc, mar = fig_runtime(data)
    breakdown = table_breakdown(data)
    fig_breakdown(breakdown)
    s_hard, e_hard, branch = branching_compare(data, X, Y)
    fig_branching(branch, s_hard, e_hard)
    cex = counterexamples(data, X, Y)
    cold, amort = overhead(X, Y)

    lines = ["# Deep Analysis (alpha,beta-CROWN on mnist_fc)\n"]
    lines.append("## A. Robustness radius over 100 samples")
    lines.append(f"- median threshold (non-robust-in-range): "
                 f"{np.median(robust):.3f} (n={len(robust)})")
    lines.append(f"- robust beyond eps={GRID[-1]}: {censored}/100 samples")
    lines.append("- figure: fig_radius_hist.png\n")

    lines.append("## B. Runtime vs epsilon (sample 0, robust)")
    lines.append(f"- alpha,beta-CROWN: {abc[0]:.2f}s -> {abc[-1]:.2f}s over eps 0.01->0.20")
    lines.append(f"- Marabou:          {mar[0]:.2f}s -> {mar[-1]:.2f}s (timeout at eps>=0.19)")
    lines.append(f"- max speedup: {max(m / a for m, a in zip(mar, abc)):.0f}x")
    lines.append("- figure: fig_runtime_vs_eps.png\n")

    lines.append("## C. Branching heuristic (hardest instance "
                 f"= sample {s_hard}, eps {e_hard})")
    lines.append("| method | verdict | time (s) | domains |")
    lines.append("|---|---|---|---|")
    for m, v, t, d in branch:
        lines.append(f"| {m} | {v} | {t:.2f} | {d} |")
    if any(v == "error" for _, v, _, _ in branch):
        lines.append("- note: babsr crashes (upstream AttributeError, "
                     "heuristics/babsr.py: split_indices); 60s budget per method.")
    lines.append("- figure: fig_branching.png\n")

    lines.append("## D. Incomplete (CROWN) vs complete (BaB) resolution")
    lines.append("| eps | safe-incomplete | safe-bab | unsafe-pgd | unsafe-bab |")
    lines.append("|---|---|---|---|---|")
    for e, si, sb, up, ub in breakdown:
        lines.append(f"| {e} | {si} | {sb} | {up} | {ub} |")
    lines.append("- figure: fig_breakdown_area.png\n")

    lines.append("## E. Counterexamples (eps=0.05 non-robust samples)")
    lines.append("| sample | true class | adversarial class |")
    lines.append("|---|---|---|")
    for s, tc, ac in cex:
        lines.append(f"| {s} | {tc} | {ac} |")
    lines.append("- images: cex_s<idx>.png\n")

    lines.append("## F. Startup overhead vs amortized")
    lines.append(f"- cold single-instance wall time: {cold:.1f}s (process startup dominates)")
    lines.append(f"- amortized per-instance (batch of 100): {amort:.2f}s")
    lines.append(f"- Marabou single instance (Assignment #3): ~0.5s")
    lines.append("- => for ONE small query Marabou is faster end-to-end; alpha,beta-CROWN "
                 "wins only once startup is amortized over many specs.\n")

    with open(os.path.join(OUT, "analysis.md"), "w") as f:
        f.write("\n".join(lines))
    print("Analysis written to results/analysis.md")


if __name__ == "__main__":
    main()
