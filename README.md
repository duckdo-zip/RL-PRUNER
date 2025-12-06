# RL-Pruner: Structured Pruning Using Reinforcement Learning for CNN Compression and Acceleration
강화학습으로 레이어별 프루닝 분포를 자동 학습하여 CNN을 구조적으로 압축 & 가속하는 방법
Residual / Concat / Flatten / SE 모듈까지 안전하게 채널을 제거하도록 텐서 의존성 그래프를 자동 구축하며, Taylor 기준으로 필터 중요도를 평가해 정확도 손실을 최소화합니다.
<br/>

Reinforcement learning is employed to automatically learn the layer-wise pruning distribution, enabling structured compression and acceleration of CNNs.
The method constructs a tensor dependency graph to safely prune channels across Residual, Concat, Flatten, and SE modules, while filter importance is estimated using the Taylor expansion criterion to minimize accuracy loss.

### Table of Contents

1. [Overview](#1️⃣-overview)
2. [PaperReview](#2️⃣-PaperReview)
3. [Method](#3️⃣-Method)
4. [Process](#4️⃣-process)
5. [Structure](#5️⃣-structure)
6. [References](#6️⃣-references)


## 1️⃣ Overview
### 1-1. 연구 배경
CNN은 높은 정확도를 가지지만 연산량·메모리 사용량이 커 엣지·실시간 적용이 어렵습니다.
구조적 프루닝(채널·필터 제거)은 하드웨어 친화적이지만 레이어별 민감도가 달라 균일 프루닝은 비효율적입니다.
따라서, 레이어별 sparsity를 자동 탐색해 정확도 손실을 최소화하는 방법이 필요하다고 생각했습니다.

<br/>

### 1-2. 핵심 기여
* RL 기반 레이어별 sparsity 학습
* 정책(policy)으로 분포를 두고 Gaussian 탐색 + Q-learning + PPO-style 업데이트
* 의존성 그래프 자동 구축
* 샘플 입력으로 텐서 흐름을 추적하여 Basic/Flatten/Concat/Residual/SE까지 채널 매핑을 자동 파악
* Taylor 기준 중요도 평가
* 가중치 × 기울기 절댓값 기반 덜 중요한 필터부터 제거하여 정확도 손실 최소화
* 연속 프루닝 + Knowledge Distillation 지원
* 단계별 프루닝 진행 후 교사 모델로 성능 회복



## 2️⃣ PaperReview
[RL-PRUNER: STRUCTURED PRUNING USING REINFORCEMENT LEARNING FOR CNN COMPRESSION AND ACCELERATION](https://arxiv.org/pdf/2411.06463) <br/>
Boyao Wang, Volodymyr Kindratenko (UIUC)


## 3️⃣ Method
### 3-1. RL 모델 파이프라인
| 항목 (Item)      | 설명 (Description)                                      |
|------------------|----------------------------------------------------------|
| **정책 (Policy)**     | 레이어별 sparsity 분포(PD)                                |
| **행동 (Action)**     | PD + Gaussian noise → 실제 sparsity 적용값                |
| **보상 (Reward)**     | 정확도 + α·FLOPs 압축 + β·파라미터 압축                    |
| **업데이트 (Update)** | Q-learning + PPO-style 클립 적용                           |
| **탐색 (Exploration)**| ε-greedy: 초기 크게 → 점진적 감소                           |

<br/>

샘플 입력을 넣어 의존성 그래프(DG) 생성
Residual, Concat, Flatten, SE까지 채널 관계 자동 파악
다단계 프루닝 단계 반복
정책 분포에서 sparsity 샘플링
Taylor 기준으로 중요도 정렬
레이어별 지정 sparsity만큼 채널 제거
보상 계산 후 정책 업데이트
필요 시 KD(knowledge distillation) 로 정확도 회복

## 4️⃣ Process
### 4-1. 데이터/모델
Datasets: CIFAR-10, CIFAR-100
Models:
VGG-19, ResNet-56, GoogLeNet, DenseNet-121, MobileNetV3-Large

<br/>

### 4-2. 전체 프루닝 흐름
초기 분포 생성 (출력 채널 기반 균일)
분포 + 노이즈 → sparsity 샘플링
의존성 그래프 따라 동시 프루닝 집합 처리
Taylor 기준으로 중요도 평가
주어진 sparsity 비율만큼 필터 제거
(옵션) KD로 성능 회복

<br/>

### 4-3. 대표 결과 요약
| 모델 (Model)                 | Sparsity      | 정확도 특징 (Accuracy Characteristics)                     |
|------------------------------|----------------|-------------------------------------------------------------|
| **VGG-19 (CIFAR-100)**       | 60%            | 정확도 하락 < 1%                                            |
| **GoogLeNet / MobileNetV3**  | 40%            | 정확도 하락 < 1%                                            |
| **ResNet-56**                | 50% 이상       | 성능 급락 (채널 수가 적은 구조적 특성)                      |
| **비교**                     | DepGraph / GReg / GNN-RL 대비 | 25%, 50%, 75% sparsity 모두 정확도 우위 |

<br/>

### 4-4. 하이퍼파라미터 예시
* Noise variance: v = 0.04
* Policy step: λ = 0.1
* Discount: γ = 0.9
* Sample num: NS = 10
* PPO clip: δ = 0.2
* Reward weight:
* 정확도: α = 0
* FLOPs: α = 0.25
* 파라미터: β = 0.25
* Exploration ε = 0.4 → cosine decay
* KD temperature τ = 0.75

### 4-5. 한계 및 주의
다단계 프루닝 + DG 생성으로 시간/자원 요구 큼
얇은 네트워크(ResNet-56 등)는 sparsity 반올림이 성능에 영향
목적(정확도 vs 속도)에 따라 α, β, ε, v 튜닝 필요

## 5️⃣ Structure
```
RL-PRUNER: STRUCTURED PRUNING USING REINFORCEMENT LEARNING FOR CNN COMPRESSION AND ACCELERATION 

├── LICENSE
├── README.md
├── code
│   ├── test
│   │   ├── test.ipynb
│   │   ├── test.ipynb  
│   │   ├── test.ipynb
│   │   ├── test.ipynb  
│   │   └── test.ipynb
│   │   
│   ├── test.py
│   ├── test.ipynb
│   └── test.py

```

## 6️⃣ References
* [RL-Pruner: Structured Pruning Using Reinforcement Learning for CNN Compression and Acceleration](https://arxiv.org/pdf/2411.06463)
