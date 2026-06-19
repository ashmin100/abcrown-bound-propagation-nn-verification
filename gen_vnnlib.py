"""Generate VNNLIB robustness specs for mnist_fc from normalized MNIST samples.

For a sample x (normalized, x_i in [-0.4242, 2.8215]) with true label L and radius eps:
  input box : x_i - eps <= X_i <= x_i + eps, clipped to the valid normalized range
  property  : robust iff L stays the argmax for every input in the box.

We encode the NEGATION (an adversarial output where some class j != L scores >= L),
so alpha,beta-CROWN reports:
  UNSAT -> property holds (verified / robust)
  SAT   -> counterexample exists (falsified)

This is the same L-inf robustness query used with Marabou in Assignment #3, so the
two tools can be compared on identical instances.
"""
import argparse
import os
import numpy as np

# Valid range of normalized MNIST: (0 - 0.1307)/0.3081 and (1 - 0.1307)/0.3081.
LO, HI = -0.4242, 2.8215


def write_vnnlib(path, x, label, eps):
    lb = np.clip(x - eps, LO, HI)
    ub = np.clip(x + eps, LO, HI)
    with open(path, "w") as f:
        f.write(f"; mnist_fc L-inf robustness | label={label} eps={eps}\n")
        for i in range(784):
            f.write(f"(declare-const X_{i} Real)\n")
        for j in range(10):
            f.write(f"(declare-const Y_{j} Real)\n")
        f.write("\n; input box (normalized space, clipped to valid range)\n")
        for i in range(784):
            f.write(f"(assert (<= X_{i} {ub[i]:.8f}))\n")
            f.write(f"(assert (>= X_{i} {lb[i]:.8f}))\n")
        f.write("\n; unsafe: some other class scores >= the true class\n")
        f.write("(assert (or\n")
        for j in range(10):
            if j != label:
                f.write(f"  (and (>= Y_{j} Y_{label}))\n")
        f.write("))\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="models/sample_inputs.npy")
    ap.add_argument("--labels", default="models/sample_labels.npy")
    ap.add_argument("--outdir", default="vnnlib")
    ap.add_argument("--epsilons", type=float, nargs="+", default=[0.01, 0.03, 0.05])
    ap.add_argument("--n", type=int, default=25, help="number of samples (from index 0)")
    args = ap.parse_args()

    X = np.load(args.inputs)
    Y = np.load(args.labels)
    os.makedirs(args.outdir, exist_ok=True)
    for eps in args.epsilons:
        for i in range(args.n):
            path = os.path.join(args.outdir, f"sample{i}_eps{eps}.vnnlib")
            write_vnnlib(path, X[i], int(Y[i]), eps)
    print(f"wrote {args.n} x {len(args.epsilons)} vnnlib files to {args.outdir}/")


if __name__ == "__main__":
    main()
