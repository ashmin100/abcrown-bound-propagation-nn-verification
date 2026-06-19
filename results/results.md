# alpha,beta-CROWN Verification Results

## Phase 1 - Sample Scan

Fixed eps = 0.05, timeout = 60s, samples 0-99.

- verified: 96, falsified: 4, timeout: 0
- non-robust samples: [8, 33, 92, 96]
- mean time: 0.206s

## Phase 2 - Epsilon Sweep

Verdict per (sample, epsilon); 'threshold' = smallest eps that is falsified.

| sample | 0.01 | 0.02 | 0.03 | 0.04 | 0.05 | 0.06 | 0.07 | 0.08 | 0.09 | 0.1 | 0.11 | 0.12 | 0.13 | 0.14 | 0.15 | 0.16 | 0.17 | 0.18 | 0.19 | 0.2 | threshold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | V | V | V | V | V | V | V | V | V | V | V | V | V | V | V | V | V | V | V | V | - |
| 8 | V | V | F | F | F | F | F | F | F | F | F | F | F | F | F | F | F | F | F | F | 0.03 |
| 33 | V | V | V | V | F | F | F | F | F | F | F | F | F | F | F | F | F | F | F | F | 0.05 |

V = verified (robust), F = falsified (counterexample), T = timeout.
