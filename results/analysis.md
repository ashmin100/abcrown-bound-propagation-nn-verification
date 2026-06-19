# Deep Analysis (alpha,beta-CROWN on mnist_fc)

## A. Robustness radius over 100 samples
- median threshold (non-robust-in-range): 0.150 (n=57)
- robust beyond eps=0.2: 43/100 samples
- figure: fig_radius_hist.png

## B. Runtime vs epsilon (sample 0, robust)
- alpha,beta-CROWN: 0.30s -> 1.48s over eps 0.01->0.20
- Marabou:          0.39s -> 300.48s (timeout at eps>=0.19)
- max speedup: 274x
- figure: fig_runtime_vs_eps.png

## C. Branching heuristic (hardest instance = sample 22, eps 0.2)
| method | verdict | time (s) | domains |
|---|---|---|---|
| babsr | error | nan | 1140 |
| fsb | timeout | 60.08 | 53696 |
| kfsb | verified | 31.30 | 26920 |
- note: babsr crashes (upstream AttributeError, heuristics/babsr.py: split_indices); 60s budget per method.
- figure: fig_branching.png

## D. Incomplete (CROWN) vs complete (BaB) resolution
| eps | safe-incomplete | safe-bab | unsafe-pgd | unsafe-bab |
|---|---|---|---|---|
| 0.01 | 100 | 0 | 0 | 0 |
| 0.02 | 100 | 0 | 0 | 0 |
| 0.03 | 99 | 0 | 1 | 0 |
| 0.04 | 96 | 1 | 3 | 0 |
| 0.05 | 95 | 1 | 4 | 0 |
| 0.06 | 94 | 1 | 5 | 0 |
| 0.07 | 92 | 2 | 5 | 0 |
| 0.08 | 87 | 5 | 8 | 0 |
| 0.09 | 86 | 5 | 9 | 0 |
| 0.1 | 78 | 12 | 10 | 0 |
| 0.11 | 68 | 19 | 13 | 0 |
| 0.12 | 57 | 28 | 15 | 0 |
| 0.13 | 43 | 39 | 18 | 0 |
| 0.14 | 35 | 41 | 23 | 0 |
| 0.15 | 31 | 40 | 29 | 0 |
| 0.16 | 23 | 45 | 32 | 0 |
| 0.17 | 17 | 44 | 39 | 0 |
| 0.18 | 11 | 41 | 47 | 0 |
| 0.19 | 7 | 39 | 53 | 0 |
| 0.2 | 5 | 38 | 57 | 0 |
- figure: fig_breakdown_area.png

## E. Counterexamples (eps=0.05 non-robust samples)
| sample | true class | adversarial class |
|---|---|---|
| 8 | 5 | 6 |
| 33 | 4 | 0 |
| 92 | 9 | 4 |
| 96 | 1 | 9 |
- images: cex_s<idx>.png

## F. Startup overhead vs amortized
- cold single-instance wall time: 2.5s (process startup dominates)
- amortized per-instance (batch of 100): 0.21s
- Marabou single instance (Assignment #3): ~0.5s
- => for ONE small query Marabou is faster end-to-end; alpha,beta-CROWN wins only once startup is amortized over many specs.
