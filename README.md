<div align="center">

# 🛡️ α,β-CROWN Robustness Verification

### Formally verifying the ℓ∞ robustness of a neural network with bound propagation — and benchmarking it against an SMT solver.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-model-005CED?logo=onnx&logoColor=white)](https://onnx.ai/)
[![α,β-CROWN](https://img.shields.io/badge/verifier-%CE%B1%2C%CE%B2--CROWN-6f42c1)](https://github.com/Verified-Intelligence/alpha-beta-CROWN)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<sub>Linear relaxation · Branch-and-bound · Complete verification · MNIST</sub>

</div>

---

## ✨ Overview

How do you *prove* that a neural network can't be fooled by a small perturbation?
This project takes a small MNIST classifier and formally verifies its **local ℓ∞
robustness** using [**α,β-CROWN**](https://github.com/Verified-Intelligence/alpha-beta-CROWN)
— the multi-time winner of the [VNN-COMP](https://sites.google.com/view/vnn2024)
neural-network verification competition — then puts the results head-to-head against
the SMT-based verifier [**Marabou**](https://github.com/NeuralNetworkVerification/Marabou).

> Two verifiers, one model, the same property — and a clear look at *why* bound
> propagation scales where SMT struggles.

## 🔍 Why this is interesting

- **Verification ≠ accuracy.** A network can be 98% accurate and still have inputs
  whose label flips under an imperceptible perturbation. Verification *proves* whether
  a robustness property holds for **every** point in an ε-ball — not just the test set.
- **Two philosophies, compared.** α,β-CROWN relaxes the network into linear bounds and
  closes the gap with branch-and-bound; Marabou encodes the network as an SMT query.
  Same question, very different engines.
- **Reproducible by design.** Pinned environment, a one-command `test.py`, and YAML
  configs you can re-run.

## 🧠 How it works

```
            input x  +  ε-ball (ℓ∞)
                     │
        ┌────────────┴─────────────┐
        │   α,β-CROWN              │   ← linear relaxation of ReLUs
        │   bound propagation     │     + α/β optimization
        │   + branch-and-bound    │     + BaB on unstable neurons
        └────────────┬─────────────┘
                     │
     ┌───────────────┼────────────────┐
  VERIFIED        FALSIFIED         TIMEOUT
 (robust ∀x)   (counterexample)   (undecided)
```

## 🚀 Getting Started

### Prerequisites
- Python 3.11 (Conda recommended)
- A clone of [α,β-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN)
  *(installed separately — not vendored in this repo)*

> Tested on macOS (Apple Silicon, arm64), Python 3.11, CPU-only.

```bash
# 1. Clone this project + the verifier (with its auto_LiRPA submodule)
git clone https://github.com/ashmin100/abcrown-bound-propagation-nn-verification.git
cd abcrown-bound-propagation-nn-verification
git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git

# 2. Create the environment and install dependencies
conda create -n abcrown python=3.11
conda activate abcrown
pip install -r requirements.txt
pip install -e alpha-beta-CROWN/auto_LiRPA      # editable install of auto_LiRPA

# 3. Sanity check (CPU)
python alpha-beta-CROWN/complete_verifier/abcrown.py \
  --config alpha-beta-CROWN/complete_verifier/exp_configs/tutorial_examples/mnist_cnn_a_adv.yaml \
  --device cpu --end 2 --timeout 30
```

> **Notes (macOS / Apple Silicon):**
> - No CUDA → always pass `--device cpu`.
> - `gurobipy` is imported by α,β-CROWN but unused here (BaB-only); no license needed.
> - If `conda` errors with a libmamba/`libarchive` message, add `--solver classic`.

### Quickstart
```bash
python test.py --config configs/mnist_fc.yaml
```

## ⚙️ Configuration

Verification runs are driven by YAML, the native α,β-CROWN format:

```yaml
model:
  onnx_path: models/mnist_fc.onnx
data:
  dataset: MNIST
specification:
  norm: .inf
  epsilon: 0.03        # ℓ∞ radius
solver:
  timeout: 60
```

## 📊 Results

> 🚧 _Populated as experiments run._

| ε | Verified | Falsified | Timeout | Avg. time |
|:---:|:---:|:---:|:---:|:---:|
| 0.01 | – | – | – | – |
| 0.03 | – | – | – | – |
| 0.05 | – | – | – | – |

## ⚔️ α,β-CROWN vs. Marabou

| | α,β-CROWN | Marabou |
|---|---|---|
| Approach | Linear relaxation + BaB | SMT (Reluplex) |
| Model format | ONNX / PyTorch | ONNX / NNet |
| Configuration | YAML | Python API |
| Speed / scalability | _TBD_ | _TBD_ |

## 🗂️ Project Structure

```
configs/        YAML verification configs
models/         ONNX model + sample inputs
test.py         minimal α,β-CROWN run on the model
results/        recorded verification results
report.pdf      analysis report
```

## 📚 References

- Wang et al., *Beta-CROWN: Efficient Bound Propagation with Per-neuron Split Constraints
  for Neural Network Robustness Verification*, NeurIPS 2021.
- Xu et al., *Fast and Complete: Enabling Complete Neural Network Verification...*, ICLR 2021.
- Katz et al., *The Marabou Framework for Verification and Analysis of DNNs*, CAV 2019.

## 📄 License

Released under the [MIT License](LICENSE).

---

<div align="center">
<sub>🇰🇷 한국어 설명은 <a href="README_ko.md">README_ko.md</a> 참고.</sub>
</div>
