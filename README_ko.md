<div align="center">

# 🛡️ α,β-CROWN 강건성 검증

### bound propagation으로 신경망의 ℓ∞ 강건성을 형식 검증하고, SMT 솔버와 비교한다.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-model-005CED?logo=onnx&logoColor=white)](https://onnx.ai/)
[![α,β-CROWN](https://img.shields.io/badge/verifier-%CE%B1%2C%CE%B2--CROWN-6f42c1)](https://github.com/Verified-Intelligence/alpha-beta-CROWN)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## ✨ 개요

신경망이 작은 섭동(perturbation)에 속지 않는다는 것을 어떻게 *증명*할까?
이 프로젝트는 소형 MNIST 분류기를 대상으로 [**α,β-CROWN**](https://github.com/Verified-Intelligence/alpha-beta-CROWN)
(VNN-COMP 신경망 검증 대회 다회 우승)으로 **국소 ℓ∞ 강건성**을 형식 검증하고,
그 결과를 SMT 기반 검증기 [**Marabou**](https://github.com/NeuralNetworkVerification/Marabou)와 비교한다.

> 같은 모델·같은 속성을 두 검증기로 — bound propagation이 SMT보다 어디서, 왜 더 잘 확장되는지 본다.

## 🔍 핵심 포인트

- **검증 ≠ 정확도.** 정확도 98%인 모델도 미세한 섭동에 라벨이 바뀌는 입력이 있을 수 있다.
  검증은 ε-공 안의 **모든** 점에 대해 속성이 성립하는지를 증명한다(테스트셋만이 아니라).
- **두 철학의 비교.** α,β-CROWN은 ReLU를 선형 완화 후 branch-and-bound로 간극을 좁히고,
  Marabou는 네트워크를 SMT 질의로 인코딩한다. 같은 질문, 전혀 다른 엔진.
- **재현 가능 설계.** 버전 고정 환경, 한 줄 `test.py`, 재실행 가능한 YAML config.

## 🚀 시작하기

### 사전 요구
- Python 3.11 (Conda 권장)
- [α,β-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN) 클론 *(이 repo에 포함하지 않음)*

> macOS(Apple Silicon, arm64), Python 3.11, CPU 환경에서 검증됨.

```bash
# 1. 본 프로젝트 + 검증기(auto_LiRPA 서브모듈 포함) 클론
git clone https://github.com/ashmin100/abcrown-bound-propagation-nn-verification.git
cd abcrown-bound-propagation-nn-verification
git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git

# 2. 환경 생성 및 의존성 설치
conda create -n abcrown python=3.11
conda activate abcrown
pip install -r requirements.txt
pip install -e alpha-beta-CROWN/auto_LiRPA      # auto_LiRPA editable 설치

# 3. 동작 확인 (CPU)
python alpha-beta-CROWN/complete_verifier/abcrown.py \
  --config alpha-beta-CROWN/complete_verifier/exp_configs/tutorial_examples/mnist_cnn_a_adv.yaml \
  --device cpu --end 2 --timeout 30
```

> **참고 (macOS / Apple Silicon):**
> - CUDA 없음 → 항상 `--device cpu`.
> - `gurobipy`는 α,β-CROWN이 import만 하고 사용하지 않음(BaB 전용) → 라이선스 불필요.
> - `conda`가 libmamba/`libarchive` 오류를 내면 `--solver classic` 추가.

### 빠른 실행
```bash
python gen_vnnlib.py        # VNNLIB 강건성 스펙 생성
python test.py              # 스모크 테스트: 두 샘플 검증(강건 1 + 비강건 1)
python run_experiments.py   # 전체 연구: 100샘플 스캔 + ε 스윕 -> results/
python analyze.py           # 심층 분석 + 그래프 -> results/
```

## ⚙️ Config

실행은 α,β-CROWN 고유 포맷인 YAML로 제어한다. 본 프로젝트는 ONNX 모델을 per-instance
VNNLIB 스펙(배치 모드, `instances.csv` 한 줄당 1개)으로 검증한다:

```yaml
general:
  device: cpu                 # CUDA 없음: CPU 실행
  csv_name: instances.csv     # 한 줄당 VNNLIB 스펙 1개 (배치)
model:
  onnx_path: models/mnist_fc.onnx
  input_shape: [-1, 784]
specification:
  vnnlib_path: null           # null => csv_name 사용
bab:
  timeout: 60                 # 인스턴스당 초
  branching:
    method: kfsb              # 분기 휴리스틱
```

VNNLIB 스펙은 ℓ∞ 공 `x_i ± ε`(유효 입력 범위로 clip)와 "정답 클래스가 arg-max를 유지"
속성을 인코딩하며, `gen_vnnlib.py`로 생성한다.

## 📊 결과

모델은 완전연결 MNIST 분류기(`784→64→32→10`, ReLU)다. 동일 인스턴스로 두 검증기를
비교하기 위해, 이전 Marabou 연구와 같은 2단계 구성을 따른다.

**Phase 1 — 샘플 스캔** (100샘플, ε=0.05): 96개 강건 검증, 4개 반증
(샘플 8, 33, 92, 96); 인스턴스당 평균 0.21초.

**Phase 2 — 강건성 임계** (반례가 처음 나오는 최소 ε):

| 샘플 | 임계 ε |
|:---:|:---:|
| 0 | 0.20까지 강건 |
| 8 | 0.03 |
| 33 | 0.05 |

### 검증 시간 vs. ε
강건 샘플에서 α,β-CROWN은 거의 일정한 반면, SMT 솔버는 가파르게 증가해 ε ≥ 0.19에서
300초 timeout에 도달한다 — 측정 구간에서 최대 약 270배 빠름.

![runtime vs epsilon](results/fig_runtime_vs_eps.png)

### 테스트셋 전체의 강건성 반경
100샘플 전부를 인증하는 비용이 작다: 43/100이 ε=0.20을 넘어서도 강건.

![robustness radius](results/fig_radius_hist.png)

### 인스턴스 해결 방식
ε가 커질수록 값싼 CROWN bound로 해결되는 비율은 줄고, branch-and-bound가 필요하거나
PGD로 반증되는 비율이 늘어난다.

![resolution breakdown](results/fig_breakdown_area.png)

### 반례
반증된 샘플에 대해 검증기는 ε-공 안의 적대적 입력을 반환한다
(원본 | 적대적 | 섭동):

![counterexample for sample 8](results/cex_s8.png)

## ⚔️ α,β-CROWN vs. Marabou

| | α,β-CROWN | Marabou |
|---|---|---|
| 접근 | 선형 완화 + branch-and-bound | SMT (Reluplex) |
| 인터페이스 | 선언형 YAML config | Python API |
| 모델 입력 | ONNX 또는 PyTorch | ONNX / NNet |
| 속성 명세 | VNNLIB 또는 dataset + ε | 코드에서 변수별 bound |
| 공통 인스턴스 판정 | Marabou와 동일 | α,β-CROWN과 동일 |
| 강건 샘플 ε=0.20 | ~1.5초 | 300초 (timeout) |
| 단일 소형 쿼리(콜드 스타트) | ~2.5초 | ~0.5초 |

두 도구가 함께 실행된 인스턴스에서 판정은 일치한다. 큰 ε에서는 α,β-CROWN이 훨씬 잘
확장되고, 단일 소형 쿼리에서는 Marabou의 시작 비용이 더 낮다.

## 🗂️ 프로젝트 구조

```
gen_vnnlib.py          VNNLIB 강건성 스펙 생성
abcrown_runner.py      α,β-CROWN 배치 실행 및 판정 파싱
test.py                최소 데모 (두 샘플)
run_experiments.py     2단계 연구 (샘플 스캔 + ε 스윕)
analyze.py             심층 분석 + 그래프
mnist_fc.yaml          검증 config
models/                ONNX 모델 + 샘플 입력
results/               표·CSV·그래프
exploration_report.pdf α,β-CROWN models/config 조사
report.pdf             분석 리포트 (영); report_ko.pdf (한)
```

## 📚 참고문헌

- Wang et al., *Beta-CROWN: Efficient Bound Propagation with Per-neuron Split Constraints
  for Neural Network Robustness Verification*, NeurIPS 2021.
- Xu et al., *Fast and Complete: Enabling Complete Neural Network Verification...*, ICLR 2021.
- Katz et al., *The Marabou Framework for Verification and Analysis of DNNs*, CAV 2019.

## 📄 라이선스

[MIT License](LICENSE).
