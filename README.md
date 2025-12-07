# Reinforcement Learning for Efficient Vision Models: From CNN Pruning to ViT Token Optimization
본 발표에서는 강화학습을 활용해 시각 모델의 효율성을 높이는 두 가지 방법을 소개합니다. **RL-Pruner**는 CNN에서 레이어별 중요도를 스스로 학습해 최적의 프루닝 비율을 찾아내며, 정확도를 유지하면서 연산량을 크게 줄입니다. **AgentViT**는 ViT에서 패치 중요도를 DDQN으로 평가해 불필요한 패치를 걸러내고, 그 결과 훈련 비용을 효과적으로 절감합니다. 두 방법 모두 “무엇을 남길지(what to keep)”를 RL이 직접 판단하도록 함으로써 모델 효율화를 자동화합니다.  
<br/>
This presentation introduces two reinforcement learning–based approaches for improving the efficiency of vision models. **RL-Pruner** learns the layer-wise importance in CNNs to determine optimal pruning ratios, significantly reducing computation while preserving accuracy. **AgentViT** evaluates patch importance in ViTs using DDQN to filter out unnecessary patches, effectively reducing training costs.  Both methods automate model optimization by allowing RL to decide “what to keep” in the network.

<img width="3200" height="1800" alt="image" src="https://github.com/user-attachments/assets/2c759310-843a-417e-a16c-796b60dd4fac" />

---

# RL-PRUNER: Structured Pruning Using Reinforcement Learning for CNN Compression and Acceleration

### Table of Contents
1. [1️⃣ Overview](#rlpruner-overview)
2. [2️⃣ Environment & Dataset](#rlpruner-envdata)
3. [3️⃣ Method](#rlpruner-method)
4. [4️⃣ Process](#rlpruner-process)
5. [5️⃣ Structure](#rlpruner-structure)
6. [6️⃣ References](#rlpruner-references)

---

# Adaptive Patch Selection to Improve Vision Transformers Through Reinforcement Learning

### Table of Contents
1. [1️⃣ Overview](#aps-overview)
2. [2️⃣ Environment & Dataset](#aps-envdata)
3. [3️⃣ Method](#aps-method)
4. [4️⃣ Process](#aps-process)
5. [5️⃣ Structure](#aps-structure)
6. [6️⃣ References](#aps-references)

---

# RL-PRUNER: Structured Pruning Using Reinforcement Learning for CNN Compression and Acceleration
RL-Pruner는 강화학습을 활용하여 CNN 레이어별 프루닝 비율을 자동으로 학습하는 구조적 프루닝 기법입니다. 먼저 텐서 흐름을 분석하여 Dependency Graph를 구축하고, 초기 sparsity 분포를 설정합니다. 이후 Gaussian noise를 적용한 다양한 action을 샘플링하여 모델을 프루닝하고, 정확도와 FLOPs·파라미터 감소율 기반 보상을 계산합니다. Replay buffer로 정책을 갱신하며, 채널 선택은 Taylor 기준을 사용합니다. 마지막으로 Knowledge Distillation으로 성능 저하를 회복합니다. 이를 통해 높은 압축률과 정확도 유지를 동시에 달성합니다.
<br/>

RL-Pruner is a structured pruning method that employs reinforcement learning to automatically learn the optimal layer-wise pruning ratios in CNNs. It first constructs a Dependency Graph by analyzing tensor flows and initializes a sparsity distribution. Then, it samples various actions by adding Gaussian noise, prunes the model accordingly, and evaluates each compressed model using rewards based on accuracy, FLOPs reduction, and parameter reduction. The policy is updated through a replay buffer, and channel selection is guided by the Taylor criterion. Finally, Knowledge Distillation is applied to recover performance degradation. This approach achieves both high compression rates and strong accuracy retention.

<img width="3200" height="1800" alt="image" src="https://github.com/user-attachments/assets/e343a86e-756a-4d44-87c0-706bb96e29bd" />

## <a id="rlpruner-overview"></a>1️⃣ Overview
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

## <a id="rlpruner-envdata"></a>2️⃣ Environment & Dataset
<img width="3200" height="1800" alt="image" src="https://github.com/user-attachments/assets/f2b3c296-688f-4134-9bad-63af94315de8" />


## <a id="rlpruner-method"></a>3️⃣ Method
### 3-1. RL 알고리즘
| 항목 (Item)      | 설명 (Description)                                      |
|------------------|----------------------------------------------------------|
| **정책 (Policy)**     | 레이어별 sparsity 분포(PD)                                |
| **행동 (Action)**     | PD + Gaussian noise → 실제 sparsity 적용값                |
| **보상 (Reward)**     | 정확도 + α·FLOPs 압축 + β·파라미터 압축                    |
| **업데이트 (Update)** | Q-learning + PPO-style 클립 적용                           |
| **탐색 (Exploration)**| ε-greedy: 초기 크게 → 점진적 감소                           |


<img width="3200" height="1800" alt="image" src="https://github.com/user-attachments/assets/f4f28cd4-a8b4-428c-b093-ce9ddff50310" />
<br/>
<br/>


<img width="3200" height="1800" alt="image" src="https://github.com/user-attachments/assets/405351e8-d044-49d1-aa54-3d3869b0dc45" />
<br/>
<br/>

샘플 입력을 넣어 의존성 그래프(DG) 생성
Residual, Concat, Flatten, SE까지 채널 관계 자동 파악
다단계 프루닝 단계 반복
정책 분포에서 sparsity 샘플링
Taylor 기준으로 중요도 정렬
레이어별 지정 sparsity만큼 채널 제거
보상 계산 후 정책 업데이트
필요 시 KD(knowledge distillation) 로 정확도 회복

## <a id="rlpruner-process"></a>4️⃣ Process
### 4-1. 전체 프루닝 흐름
초기 분포 생성 (출력 채널 기반 균일)
분포 + 노이즈 → sparsity 샘플링
의존성 그래프 따라 동시 프루닝 집합 처리
Taylor 기준으로 중요도 평가
주어진 sparsity 비율만큼 필터 제거
(옵션) KD로 성능 회복
<br/>

<img width="3200" height="1800" alt="image" src="https://github.com/user-attachments/assets/d5e2efe6-02bd-41a5-ad9f-bcd8d9e251ec" />
<img width="3200" height="1800" alt="image" src="https://github.com/user-attachments/assets/6eb1868e-57e5-4d50-9a62-fa556234fc80" />
<img width="3200" height="1800" alt="image" src="https://github.com/user-attachments/assets/2751168d-fd14-4565-b611-146435fe1fa4" />


### 4-2. 대표 결과 요약
| 모델 (Model)                 | Sparsity      | 정확도 특징 (Accuracy Characteristics)                     |
|------------------------------|----------------|-------------------------------------------------------------|
| **VGG-19 (CIFAR-100)**       | 60%            | 정확도 하락 < 1%                                            |
| **GoogLeNet / MobileNetV3**  | 40%            | 정확도 하락 < 1%                                            |
| **ResNet-56**                | 50% 이상       | 성능 급락 (채널 수가 적은 구조적 특성)                      |
| **비교**                     | DepGraph / GReg / GNN-RL 대비 | 25%, 50%, 75% sparsity 모두 정확도 우위 |
<br/>

### 4-3. 하이퍼파라미터 예시
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
<br/>

### 4-4. 한계 및 주의
다단계 프루닝 + DG 생성으로 시간/자원 요구 큼
얇은 네트워크(ResNet-56 등)는 sparsity 반올림이 성능에 영향
목적(정확도 vs 속도)에 따라 α, β, ε, v 튜닝 필요

<img width="3200" height="1800" alt="image" src="https://github.com/user-attachments/assets/9e79e0a9-257b-4012-84d7-5171dd66f29f" />


## <a id="rlpruner-structure"></a>5️⃣ Structure
```
RL-PRUNER: STRUCTURED PRUNING USING REINFORCEMENT LEARNING FOR CNN COMPRESSION AND ACCELERATION 

RLPruner-CNN/
├── assets/                         # 그림, 실험 결과, 문서 자료
│   ├── CNN_method_description.jpg
│   ├── experiments_result.jpg
│   └── README.md
│
├── checkpoint/                     # 모델 체크포인트 저장 경로
│
├── compressed_model/               # 압축(Pruned)된 모델 저장 경로
│
├── conf/                           # 환경 설정 파일들
│   ├── __init__.py
│   └── global_settings.py
│
├── data/                           # CIFAR 데이터셋
│   ├── cifar-100-python/
│   └── cifar-100-python.tar.gz
│
├── log/                            # 학습 로그 및 wandb 로그 등이 저장되는 폴더
│
├── models/                         # 모델 아키텍처 모음
│   ├── __pycache__/
│   ├── densenet.py
│   ├── googlenet.py
│   ├── mobilenentv3.py
│   ├── resnet_tiny.py
│   ├── resnet.py
│   └── vgg.py
│
├── pretrained_model/               # 사전학습(pretrained) 모델 파일들
│   ├── googlenet_cifar100_pretrained.pth
│   ├── mobilenetv3_large_cifar100_pretrained.pth
│   ├── resnet56_cifar100_pretrained.pth
│   └── vgg19_cifar100_pretrained.pth
│
├── scripts/                        # 실행 스크립트 (.sh)
│   ├── evaluate.sh
│   ├── example.sh
│   ├── flexible.sh
│   ├── prune.sh
│   └── train.sh
│
├── utils/                          # 유틸리티 모듈
│   └── wandb/                      # wandb 실행 관련 서브 폴더
│
├── avg_prune_prob_vgg19.png
├── avg_prune_prob.png
├── compress.py                     # 모델 압축(Pruning) 메인 코드
├── evaluate.py                     # 모델 평가
├── requirements.txt
└── train.py                        # 학습 엔트리 포인트
```

## <a id="rlpruner-references"></a>6️⃣ References
* [RL-Pruner: Structured Pruning Using Reinforcement Learning for CNN Compression and Acceleration](https://arxiv.org/pdf/2411.06463)
* [https://github.com/Beryex/RLPruner-CNN](https://github.com/Beryex/RLPruner-CNN)

---

# Adaptive Patch Selection to Improve Vision Transformers Through Reinforcement Learning
AgentViT는 ViT의 첫 attention 값을 상태로 사용해 DDQN 에이전트가 중요 패치를 선택하는 구조적 프루닝 프레임워크입니다. 선택된 패치만 후속 레이어에 전달해 연산량을 줄이고, 보상(손실·패치 수)을 통해 정책을 최적화했습니다. CIFAR10·FMNIST·Imagenette+ 실험에서 정확도 유지 또는 향상과 함께 학습시간·GFLOPs·FPS 개선을 확인했습니다.
<br/>

AgentViT is a structural pruning framework where a DDQN agent selects important patches using the first-layer attention of a ViT as the state. Only selected patches are passed to later layers to reduce computation, and the agent’s policy is optimized via rewards based on loss and patch count. Experiments on CIFAR-10, FMNIST, and Imagenette+ show maintained or improved accuracy with reduced training time, GFLOPs, and increased FPS.


<img width="3200" height="1800" alt="image" src="https://github.com/user-attachments/assets/ddf1c3ad-6e4f-420b-b589-76d0c17ee88d" />


## <a id="aps-overview"></a>1️⃣ Overview
내용 작성…

## <a id="aps-envdata"></a>2️⃣ Environment & Dataset
<img width="3200" height="1800" alt="image" src="https://github.com/user-attachments/assets/d7d2dd0a-f35e-4a7d-9018-84f75271271d" />


## <a id="aps-method"></a>3️⃣ Method
내용 작성…

## <a id="aps-process"></a>4️⃣ Process
<img width="3200" height="1800" alt="image" src="https://github.com/user-attachments/assets/f1bfdd96-cb8a-411d-826b-908398d91571" />
<img width="3200" height="1800" alt="image" src="https://github.com/user-attachments/assets/57ef8b4c-ac4e-4b47-88b3-9d4f486de39c" />
<img width="3200" height="1800" alt="image" src="https://github.com/user-attachments/assets/157c62c3-9e72-4e7a-918f-0fd7bad11513" />
<img width="3200" height="1800" alt="image" src="https://github.com/user-attachments/assets/fff972d4-0430-4b44-ab6e-ce122ba2318f" />


## <a id="aps-structure"></a>5️⃣ Structure
<img width="3200" height="1800" alt="image" src="https://github.com/user-attachments/assets/f13162c8-caa5-43bd-896e-569070cf4a89" />


## <a id="aps-references"></a>6️⃣ References
* [Adaptive patch selection to improve Vision Transformers through Reinforcement Learning](https://link.springer.com/article/10.1007/s10489-025-06516-z)
* [https://github.com/DavideTraini/RL-for-ViT](https://github.com/DavideTraini/RL-for-ViT)
