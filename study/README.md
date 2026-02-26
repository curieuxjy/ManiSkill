# Allegro Hand 스터디

ManiSkill의 Allegro Hand 관련 코드를 학습하기 위한 작업 폴더.

> **원칙:** ManiSkill 메인 소스(`mani_skill/`)는 일절 수정하지 않습니다.
> 원본 환경의 버그는 상속 + 오버라이드로 수정한 커스텀 환경을 등록하여 사용합니다.

---

## Step 0. 시스템 요구사항

| 항목 | 요구사항 |
|------|----------|
| OS | **Linux + NVIDIA GPU** 권장 (GPU 시뮬레이션은 Linux만 지원) |
| Python | 3.9 이상, **3.11 권장** |
| GPU | NVIDIA GPU + 최신 드라이버 |
| Vulkan | 렌더링에 필요 (GPU 시뮬레이션과 별개) |

> macOS/Windows에서는 CPU 시뮬레이션(단일 환경)만 가능하고 GPU 병렬 시뮬레이션은 불가합니다.

## Step 1. 환경 세팅 및 설치

### 1-1. Python 가상환경 생성

**uv 사용 (권장):**
```bash
git clone https://github.com/haosulab/ManiSkill.git
cd ManiSkill
uv venv -p python3.11
source .venv/bin/activate
```

**conda 사용:**
```bash
conda create -n ms3 "python=3.11"
conda activate ms3
git clone https://github.com/haosulab/ManiSkill.git
cd ManiSkill
```

### 1-2. ManiSkill + PyTorch 설치

> **주의:** `uv venv`로 만든 환경에서는 `pip`이 아니라 `uv pip`을 사용해야 합니다.

```bash
# 소스에서 editable 설치 (코드를 읽으며 공부할 때 유리)
uv pip install -e .

# PyTorch (CUDA 버전에 맞춰 설치)
# https://pytorch.org/get-started/locally/ 참고
uv pip install torch
```

### 1-3. 추가 의존성 설치

```bash
# SAPIEN이 사용하는 pkg_resources가 setuptools>=82에서 제거됨 — 반드시 82 미만으로 고정
uv pip install "setuptools<82"

# PPO/SAC 학습 스크립트에 필요
uv pip install tensorboard

# 실험 추적 (wandb)
uv pip install wandb
wandb login
```

빠뜨리면 각각 `ModuleNotFoundError: No module named 'pkg_resources'`, `No module named 'tensorboard'` 에러가 발생합니다.

### 1-4. Vulkan 설치 (렌더링용)

```bash
# Ubuntu
sudo apt-get install libvulkan1

# 설치 확인
vulkaninfo | head -20
```

`vulkaninfo`가 실패하면 `/usr/share/vulkan/icd.d/nvidia_icd.json` 파일이 있는지 확인하세요. 없으면 NVIDIA 드라이버를 재설치해야 합니다:
```bash
# 드라이버 확인
ldconfig -p | grep libGLX_nvidia

# 없으면 드라이버 재설치 (xxx는 버전)
sudo apt-get install nvidia-driver-xxx
```

### 1-5. 설치 확인

```bash
# 기본 동작 테스트 (렌더링 없이)
python -m mani_skill.examples.demo_random_action

# GUI 뷰어 테스트 (디스플레이 있을 때)
python -m mani_skill.examples.demo_random_action -e PickCube-v1 --render-mode="human"

# GPU 시뮬레이션 테스트
python -m mani_skill.examples.benchmarking.gpu_sim --num-envs=64
```

세 명령 모두 에러 없이 실행되면 설치 완료입니다.

## Step 2. Allegro Hand 환경 확인

```bash
# 관찰/행동 공간 구조 출력 + 랜덤 스텝 실행
python study/scripts/01_explore_env.py
```

이 스크립트가 출력하는 내용:
- Action space: `Box(-1, 1, (16,))` — 16개 관절의 delta position
- Observation 딕셔너리 구조: qpos, qvel, palm_pose, tip_poses, fsr_impulse 등
- 10스텝 랜덤 행동의 reward와 success/fail 상태

## Step 3. 시각적 확인

```bash
# 비디오 저장 (디스플레이 없어도 됨)
python study/scripts/02_visualize.py --mode video

# GUI 뷰어 (디스플레이 필요)
python study/scripts/02_visualize.py --mode viewer
```

## Step 4. 커스텀 환경 (`AllegroRotateLevel{0-3}-v1`)

### 왜 커스텀 환경을 만들었는가

ManiSkill 원본 환경 `RotateSingleObjectInHandLevel{0-3}-v1`에는 **partial reset 버그**가 있습니다:

```
원본 코드 (rotate_single_object_in_hand.py:170-172):

new_pos[:, 2] = torch.abs(new_pos[:, 2]) + self.hand_init_height + self.obj_heights
                ~~~~~~~~~~~~~~~~~~~~~~~~~~~                          ~~~~~~~~~~~~~~~~
                shape: [len(env_idx)]                                shape: [num_envs]
                (리셋 대상 환경 수)                                   (전체 환경 수)
```

GPU 병렬 시뮬레이션에서 일부 환경만 리셋(partial reset)될 때, `new_pos`는 리셋 대상 환경 수만큼의
크기를 갖지만 `self.obj_heights`는 전체 환경 수 크기이므로 **텐서 크기 불일치 에러**가 발생합니다.
(Level 0은 `obj_heights`가 1개라 브로드캐스트로 우연히 동작하지만, Level 1+에서 터짐)

### 해결 방식: 상속 + 오버라이드

메인 소스를 수정하는 대신, 원본 클래스를 상속하여 버그가 있는 메서드만 오버라이드한
**새로운 환경을 등록**합니다:

```
study/envs/allegro_rotate.py

  RotateSingleObjectInHand          ← ManiSkill 원본 (수정 안 함)
    └── AllegroRotate               ← _initialize_actors만 오버라이드 (버그 수정)
          ├── AllegroRotateLevel0   ← @register_env("AllegroRotateLevel0-v1")
          ├── AllegroRotateLevel1   ← @register_env("AllegroRotateLevel1-v1")
          ├── AllegroRotateLevel2   ← @register_env("AllegroRotateLevel2-v1")
          └── AllegroRotateLevel3   ← @register_env("AllegroRotateLevel3-v1")
```

수정 내용은 `_initialize_actors` 메서드의 한 줄뿐입니다:

```python
# 수정: obj_heights가 1개(Level 0)이면 브로드캐스트, 여러 개(Level 1+)이면 슬라이싱
obj_h = self.obj_heights if len(self.obj_heights) == 1 else self.obj_heights[env_idx]
new_pos[:, 2] = torch.abs(new_pos[:, 2]) + self.hand_init_height + obj_h
```

나머지 모든 기능(씬 로드, 보상, 관찰, 평가)은 원본을 그대로 상속합니다.

### 환경 이름 매핑

| 원본 (버그 있음) | 커스텀 (수정됨) | 비고 |
|---|---|---|
| `RotateSingleObjectInHandLevel0-v1` | `AllegroRotateLevel0-v1` | 고정 큐브 |
| `RotateSingleObjectInHandLevel1-v1` | `AllegroRotateLevel1-v1` | 랜덤 큐브 |
| `RotateSingleObjectInHandLevel2-v1` | `AllegroRotateLevel2-v1` | YCB, z축 |
| `RotateSingleObjectInHandLevel3-v1` | `AllegroRotateLevel3-v1` | YCB, 랜덤축 |

### 실행 구조: `run_with_patch.py`

`ppo.py`/`sac.py`는 ManiSkill 내장 환경만 등록(`import mani_skill.envs`)합니다.
커스텀 환경을 인식시키려면 `allegro_rotate.py`가 먼저 import되어야 합니다.

`run_with_patch.py`가 이를 자동으로 처리합니다:

```
run_with_patch.py가 하는 일:
  1. 대상 스크립트(ppo.py 등)를 읽는다
  2. import mani_skill.envs 직후에 커스텀 환경 등록 코드를 주입한다
  3. (SAC인 경우) eval_metrics KeyError 수정도 함께 적용한다
  4. 수정된 스크립트를 실행한다
```

셸 스크립트에서의 사용:

```bash
# 직접 실행 (커스텀 환경 인식 안 됨)
python examples/baselines/ppo/ppo.py --env_id=AllegroRotateLevel0-v1  # ✗ 에러

# run_with_patch.py 경유 (커스텀 환경 자동 등록)
python study/scripts/run_with_patch.py \
    examples/baselines/ppo/ppo.py --env_id=AllegroRotateLevel0-v1     # ✓ 동작
```

모든 학습/평가 셸 스크립트(`03~06`)는 이미 `run_with_patch.py`를 경유하도록 설정되어 있으므로,
셸 스크립트로 실행하면 별도 설정 없이 바로 동작합니다.

## Step 5. 학습 (PPO)

```bash
# Level 0 (고정 크기 큐브) — 가장 쉬운 태스크
bash study/scripts/03_train_ppo.sh 0 512 5000000
```

기본으로 wandb에 로깅됩니다 (프로젝트: `ManiSkill3_AllegroHand`). wandb 없이 실행하려면:

```bash
WANDB=0 bash study/scripts/03_train_ppo.sh 0 512 5000000
```

학습 중 모니터링:

```bash
# TensorBoard (로컬) — 로그가 study/runs/ 하위에 저장됨
tensorboard --logdir=study/runs/
# 브라우저에서 http://localhost:6006

# WandB — wandb.ai 대시보드에서 ManiSkill3_AllegroHand 프로젝트 확인
```

**GPU 메모리에 따른 `num_envs` 조절:**
- 8GB VRAM → `--num_envs=64`
- 16GB VRAM → `--num_envs=256`
- 24GB+ VRAM → `--num_envs=512`

학습 결과(체크포인트, TensorBoard 로그, wandb 로그)는 모두 `study/runs/`와 `study/wandb/` 하위에 저장됩니다.

## Step 6. 학습 (SAC)

SAC는 PPO보다 sample-efficient하지만 wall-clock 시간은 더 걸릴 수 있습니다.

```bash
bash study/scripts/04_train_sac.sh 0 64 2000000
```

## Step 7. 평가

PPO와 SAC는 체크포인트 형식이 다르므로 알고리즘을 지정해야 합니다.

```bash
# PPO 모델 평가 (체크포인트 경로는 study/ 기준 상대경로)
bash study/scripts/05_evaluate.sh ppo runs/allegro_ppo_.../final_ckpt.pt 0

# SAC 모델 평가
bash study/scripts/05_evaluate.sh sac runs/allegro_sac_.../ckpt_50000.pt 0
```

## Step 8. 난이도 올리기

| Level | 물체 | 회전축 | 추가 에셋 |
|-------|------|--------|-----------|
| 0 | 고정 크기 큐브 (half_size=0.04) | z축 고정 | 없음 |
| 1 | 랜덤 크기 큐브 | z축 고정 | 없음 |
| 2 | YCB 실물체 (머그컵, 바나나 등) | z축 고정 | `ycb` |
| 3 | YCB 실물체 | x/y/z 랜덤 | `ycb` |

### YCB 에셋 동작 방식

Level 2/3에서는 큐브 대신 [YCB 데이터셋](https://www.ycbbenchmarks.com/)의 실제 물체(머그컵, 바나나, 스패너 등)가 사용됩니다.
물체마다 형상과 질량이 다르기 때문에 Level 0/1보다 훨씬 어렵습니다.

**자동 다운로드:** Level 2/3 환경을 처음 만들면 ManiSkill이 `~/.maniskill/data/`에 YCB 에셋이 있는지 확인하고,
없으면 HuggingFace에서 자동 다운로드합니다. 미리 받아두려면:

```bash
python -m mani_skill.utils.download_asset ycb
```

**코드상 동작** (`rotate_single_object_in_hand.py` 125~139행):

```python
# difficulty_level >= 2 일 때 (_load_scene 내부)

# 1. 사용 가능한 YCB 모델 ID 목록 로드
all_model_ids = load_json(ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json").keys()

# 2. 환경마다 랜덤으로 물체 하나 선택
model_ids = self._batched_episode_rng.choice(all_model_ids)

# 3. "ycb:{model_id}" 형식으로 메시+충돌체 로드
builder = actors.get_actor_builder(self.scene, id=f"ycb:{model_id}")
```

**Level 2 vs 3 차이** (`_initialize_actors` 181~184행) — 유일한 차이는 회전 목표축:

```python
if self.difficulty_level <= 2:
    axis = torch.ones((b,), dtype=torch.long) * 2        # z축 고정
else:
    axis = torch.randint(0, 3, (b,), dtype=torch.long)   # x/y/z 랜덤
```

### 방법 A. 레벨별 독립 학습

각 레벨을 처음부터 따로 학습합니다:

```bash
bash study/scripts/03_train_ppo.sh 0 256 5000000
bash study/scripts/03_train_ppo.sh 1 256 10000000
bash study/scripts/03_train_ppo.sh 2 256 10000000
bash study/scripts/03_train_ppo.sh 3 256 15000000
```

### 방법 B. 커리큘럼 학습 (권장)

이전 레벨에서 학습한 모델을 다음 레벨의 초기 가중치로 사용합니다.
난이도가 올라갈수록 learning rate를 낮춰서 기존에 학습한 정책을 크게 흐트러뜨리지 않습니다.

```
Level 0 (lr=3e-4, 5M steps)
  → checkpoint를 Level 1 초기 가중치로
Level 1 (lr=1e-4, 10M steps)
  → checkpoint를 Level 2 초기 가중치로
Level 2 (lr=1e-4, 10M steps)
  → checkpoint를 Level 3 초기 가중치로
Level 3 (lr=5e-5, 15M steps)
```

전체 자동 실행:

```bash
bash study/scripts/06_curriculum_train.sh
```

#### 빠른 검증 (레벨업 파이프라인 테스트)

체크포인트 전달이 정상 동작하는지만 확인하려면 축소 버전을 사용합니다:

```bash
bash study/scripts/06_curriculum_quick_test.sh
```

| 항목 | 본 학습 (`06_curriculum_train.sh`) | 빠른 검증 (`06_curriculum_quick_test.sh`) |
|---|---|---|
| `num_envs` | 1024 | 64 |
| `total_timesteps` | 5천만~15억 | 50만 (전 레벨 동일) |
| `update_epochs` | 8 | 4 |
| `num_minibatches` | 32 | 8 |
| wandb / 영상 | O | X |

또는 수동으로 한 단계씩:

```bash
# Level 0
bash study/scripts/03_train_ppo.sh 0 256 5000000

# Level 1 — Level 0 체크포인트에서 이어서 (run_with_patch.py 경유)
cd study
python scripts/run_with_patch.py ../examples/baselines/ppo/ppo.py \
  --env_id="AllegroRotateLevel1-v1" \
  --checkpoint=runs/<level0_exp>/final_ckpt.pt \
  --num_envs=256 --num_steps=50 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=10000000 --eval_freq=25 --gamma=0.8 \
  --learning_rate=1e-4 --control_mode="pd_joint_delta_pos" --save_model
```

핵심은 `--checkpoint=<이전 레벨 체크포인트>` 옵션으로 가중치를 이어받는 것입니다.

---

## ManiSkill3 GPU 병렬화 심화 학습

ManiSkill3의 세 가지 핵심 GPU 병렬화 기능을 체험하는 스크립트:

### 08. GPU 비주얼 데이터 수집

RGBD + Segmentation 데이터를 GPU에서 병렬 렌더링하여 30,000+ FPS로 수집:

```bash
python study/scripts/08_gpu_visual_data.py                      # 전체 실행
python study/scripts/08_gpu_visual_data.py --section explore     # obs_mode별 데이터 구조
python study/scripts/08_gpu_visual_data.py --section benchmark   # FPS 벤치마크
python study/scripts/08_gpu_visual_data.py --section save        # 샘플 이미지 저장
```

### 09. GPU 병렬 시뮬레이션 & 상태 수집

PhysX GPU 백엔드로 수천 개 환경의 상태를 배치 텐서로 수집/복원:

```bash
python study/scripts/09_gpu_state_collection.py                      # 전체 실행
python study/scripts/09_gpu_state_collection.py --section backend    # CPU vs GPU 비교
python study/scripts/09_gpu_state_collection.py --section state      # 상태 수집/복원
python study/scripts/09_gpu_state_collection.py --section benchmark  # FPS 벤치마크
python study/scripts/09_gpu_state_collection.py --section collect    # 대량 데이터 수집
```

### 10. GPU 이종(heterogeneous) 시뮬레이션

각 병렬 환경이 완전히 다른 물체/씬을 가지는 이종 환경 학습:

```bash
python study/scripts/10_gpu_heterogeneous.py                         # 전체 실행
python study/scripts/10_gpu_heterogeneous.py --section scene_idx     # set_scene_idxs 원리
python study/scripts/10_gpu_heterogeneous.py --section merge         # Actor.merge 구조
python study/scripts/10_gpu_heterogeneous.py --section hetero_env    # 이종 환경 실제 예시
python study/scripts/10_gpu_heterogeneous.py --section reconfig      # reconfiguration_freq
```

---

## 관련 소스 코드 맵

### study 구조

```
study/
├── envs/
│   └── allegro_rotate.py          ← 커스텀 환경 (원본 상속 + 버그 수정)
├── scripts/
│   ├── run_with_patch.py          ← 커스텀 환경 등록 주입 래퍼
│   ├── 01_explore_env.py          ← 환경 탐색
│   ├── 02_visualize.py            ← 시각화 (비디오/GUI)
│   ├── 03_train_ppo.sh            ← PPO 학습
│   ├── 04_train_sac.sh            ← SAC 학습
│   ├── 05_evaluate.sh             ← 평가
│   ├── 06_curriculum_train.sh     ← 커리큘럼 학습 (Level 0→3)
│   ├── 06_curriculum_quick_test.sh← 커리큘럼 레벨업 빠른 검증
│   ├── 07_parallel_viewer.py      ← 병렬 환경 시각화
│   ├── 08_gpu_visual_data.py      ← GPU 비주얼 데이터 수집 학습
│   ├── 09_gpu_state_collection.py ← GPU 상태 수집 학습
│   └── 10_gpu_heterogeneous.py    ← GPU 이종 시뮬레이션 학습
├── runs/                          ← 체크포인트, TensorBoard 로그 (gitignore)
├── wandb/                         ← WandB 로그 (gitignore)
└── README.md
```

### ManiSkill 원본 소스

| 분류 | 경로 | 설명 |
|------|------|------|
| 로봇 에이전트 | `mani_skill/agents/robots/allegro_hand/allegro.py` | AllegroHandRight/Left (16 DOF) |
| 로봇 에이전트 | `mani_skill/agents/robots/allegro_hand/allegro_touch.py` | AllegroHandRightTouch (FSR 촉각 16개) |
| 태스크 환경 | `mani_skill/envs/tasks/dexterity/rotate_single_object_in_hand.py` | 물체 회전 원본 (Level 0~3) |
| URDF/에셋 | `mani_skill/assets/robots/allegro/` | URDF 파일 및 메시 |
| PPO | `examples/baselines/ppo/ppo.py` | PPO 학습 스크립트 |
| SAC | `examples/baselines/sac/sac.py` | SAC 학습 스크립트 |

## 환경 구조 이해

### 클래스 계층

```
BaseEnv (mani_skill/envs/sapien_env.py)
  └── RotateSingleObjectInHand           ← 원본 base (partial reset 버그 있음)
        ├── RotateSingleObjectInHandLevel0-v1  ← 원본 등록 환경
        ├── RotateSingleObjectInHandLevel1-v1
        ├── ...
        └── AllegroRotate                ← study 커스텀 (_initialize_actors 오버라이드)
              ├── AllegroRotateLevel0-v1  ← 커스텀 등록 환경 (학습에 사용)
              ├── AllegroRotateLevel1-v1
              ├── AllegroRotateLevel2-v1
              └── AllegroRotateLevel3-v1
```

### 핵심 흐름

```
gym.make("AllegroRotateLevel0-v1")
  → AllegroRotateLevel0 (AllegroRotate 상속)
    → robot_uids="allegro_hand_right_touch" (촉각 센서 포함)
    → _load_scene(): 테이블 + 큐브/YCB 물체 생성          ← 원본 그대로
    → _initialize_actors(): 물체 초기 위치, 회전 목표축    ← 오버라이드 (버그 수정)
    → _initialize_agent(): 로봇 초기 자세                  ← 원본 그대로
    → evaluate(): 누적 회전각, 낙하 여부, 성공 판단        ← 원본 그대로
    → compute_dense_reward(): 보상 계산                    ← 원본 그대로
```

### 관찰 / 행동 / 보상

**관찰(Observation)** 구성:
- 로봇 proprioception: qpos, qvel, palm_pose, tip_poses (4개 손가락 끝), fsr_impulse (16개 촉각)
- 추가 obs: rotate_dir (목표 회전축), obj_pose, obj_tip_vec (state 모드일 때)

**행동(Action)**: 16차원 (pd_joint_delta_pos 기본) — 각 관절의 위치 변화량

**보상(Reward)**:
- `+20 × 회전각`: 목표 방향 회전 보상
- `-0.1 × 물체 속도`: 안정성
- `-50 × 낙하`: 물체 떨어뜨리면 큰 페널티
- `-0.0003 × (power + torque)`: 에너지 효율
- `+거리 보상`: 손가락이 물체에 가까울수록

**성공 조건**: 누적 회전각 > 4π (약 720°)
