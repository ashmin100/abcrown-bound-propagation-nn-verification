# Exploration Report — α,β-CROWN Models & Configuration System

This report documents the α,β-CROWN repository's built-in models directory and
configuration system (Assignment Problem 1). All figures below were obtained by
inspecting the cloned repository at `complete_verifier/` and are cross-checked against
the actual files.

## 1. What models are provided (architecture, format)?

The built-in models live under `complete_verifier/models/`, organized into 13
category directories:

```
bab_attack  cifar10_resnet  crown-ibp  custom_op  custom_specs  eran
l2_norm     marabou_cifar10 non_relu   oval       sdp           toy   vnncomp22
```

**Format.** Models are stored as **PyTorch checkpoints**, not ONNX:

| Extension | Count | Meaning |
|---|---|---|
| `.pth` | 24 | PyTorch `state_dict` checkpoints |
| `.model` | 8 | PyTorch checkpoints (SDP benchmark naming) |
| `.pkl` | 4 | pickled weights |
| `.pt` | 1 | PyTorch checkpoint |
| `.onnx` | **0** | (none in `models/`) |

There are **no ONNX files** in the models directory. The checkpoints only hold weights;
the matching network *architectures* are defined as ~97 constructor functions in
`model_defs.py`. A config selects an architecture by `name:` and loads the weights from
`path:`. ONNX is supported for *external* models (via `onnx_path:`), but the bundled
collection is PyTorch-based.

**Architecture mix** (approximate, by function-name keyword in `model_defs.py`):
~35 ResNet variants, ~29 CNN/conv networks, ~21 fully-connected/MLP networks, and no
recurrent/transformer models. In short, the suite is dominated by small-to-medium
**CIFAR-10 and MNIST image classifiers** drawn from verification benchmarks (ERAN, SDP,
OVAL, CROWN-IBP, VNN-COMP).

## 2. What verification configurations (YAML files) are available?

α,β-CROWN is driven entirely by YAML. `complete_verifier/exp_configs/` contains
**263 YAML files** across 10 directories:

| Directory | YAML files | Theme |
|---|---|---|
| BICCOS | 128 | BICCOS cutting-plane experiments |
| beta_crown | 27 | β-CROWN benchmarks |
| vnncomp22 | 27 | VNN-COMP 2022 |
| tutorial_examples | 18 | minimal worked examples |
| vnncomp21 | 14 | VNN-COMP 2021 |
| GCP-CROWN | 13 | GCP-CROWN experiments |
| vnncomp23 | 13 | VNN-COMP 2023 |
| vnncomp25 | 9 | VNN-COMP 2025 |
| vnncomp24 | 8 | VNN-COMP 2024 |
| bab_attack | 6 | falsification/attack |

So configs are organized by **competition year** and by **algorithmic method**, plus a
tutorial set. A config has **9 top-level sections** (verified from a runtime config dump):
`general, model, data, specification, solver, bab, attack, debug, solving`, exposing
**308 options** in total. The most relevant keys:

- `model` — `name` (architecture from `model_defs.py`) + `path`, or `onnx_path`; `input_shape`.
- `data` — `dataset`, normalization `mean`/`std`, `start`/`end` sample indices.
- `specification` — `norm` (L-inf / L2 / L1), `epsilon`, or a `vnnlib_path`.
- `solver` — `batch_size`, `alpha-crown`/`beta-crown` learning rates and iterations.
- `bab` — `timeout`, `branching` (`method` = babsr/fsb/kfsb, `reduceop`, `candidates`).
- `attack` — PGD settings (`pgd_order`, `pgd_steps`, `pgd_restarts`).

**Properties.** The verification property is almost always an L-infinity robustness ball:
of the configs that set a norm, **140 use `.inf`** and 2 use L2. The perturbed region is
given either as `epsilon` over a built-in dataset, or as an explicit **VNNLIB** file.
**Datasets** used are mostly CIFAR-10 and MNIST variants (e.g. `CIFAR_SDP` ×60,
`CIFAR` ×32, `MNIST_SDP` ×10, plus `_ERAN` normalization variants, `CIFAR100`, and
`Customized`).

## 3. How does α,β-CROWN's model specification differ from Marabou's?

| Aspect | α,β-CROWN | Marabou (Assignment #3) |
|---|---|---|
| Interface | Declarative **YAML config** file | Imperative **Python API** (`maraboupy`) |
| Model load | `name:`+`path:` (PyTorch arch, used by 113 configs) or `onnx_path:` (4) | `Marabou.read_onnx(path)` |
| Input region | `epsilon` over a dataset, or a `vnnlib_path` (VNNLIB) | per-variable `setLowerBound`/`setUpperBound` in a loop |
| Output property | dataset auto-generates the robustness spec, or VNNLIB encodes it | hand-built `MarabouUtils.Equation` constraints |
| Batch runs | one CSV lists many (model, spec) instances → one process | one script per query |
| Solver control | config keys (branching, α/β iterations, PGD, timeout) | a few API options |

In practice, α,β-CROWN separates *what to verify* (declarative config / VNNLIB) from
*the code*, so a new experiment is a new YAML file. Marabou builds the query
**programmatically**: the model is read with `read_onnx`, the input box is set variable
by variable, and the output property is assembled from `Equation` objects in Python.

## Organization compared with Marabou's resources

α,β-CROWN ships a large, curated benchmark suite — a models directory grouped by
benchmark source plus 263 ready-to-run configs organized by competition year and method.
Marabou is closer to a verification *library*: it provides the API and a few example
networks, and the user writes the query script themselves. The trade-off is convenience
and reproducibility (α,β-CROWN's config/VNNLIB ecosystem) versus flexibility (Marabou's
direct programmatic control).
