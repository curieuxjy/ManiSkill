#!/bin/bash
# Allegro Hand 커리큘럼 학습 스크립트
#
# 낮은 레벨에서 학습한 모델을 다음 레벨의 초기 가중치로 사용합니다.
#
# 사용법:
#   bash study_allegro/scripts/06_curriculum_train.sh
#
# 각 레벨의 timestep과 num_envs는 필요에 따라 아래에서 조정하세요.

set -e

# 프로젝트 루트와 study_allegro 디렉토리 경로 설정
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STUDY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$STUDY_DIR/.." && pwd)"

NUM_ENVS=1024
TAG=$(date +%Y%m%d_%H%M%S)

# study_allegro/ 하위에 runs/, wandb/ 저장
cd "${STUDY_DIR}"
export WANDB_DIR="${STUDY_DIR}"

PATCH="${STUDY_DIR}/scripts/run_with_patch.py"
PPO_SCRIPT="${PROJECT_ROOT}/examples/baselines/ppo/ppo.py"

# ──────────────────────────────────────────────
# Level 0: 고정 크기 큐브
# ──────────────────────────────────────────────
EXP_L0="allegro_curriculum_level0_${TAG}"
echo "=== Level 0 학습 시작 ==="
python "${PATCH}" "${PPO_SCRIPT}" \
    --env_id="AllegroRotateLevel0-v1" \
    --exp_name="${EXP_L0}" \
    --num_envs=${NUM_ENVS} --num_steps=50 \
    --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50000000 --eval_freq=25 \
    --gamma=0.8 --gae_lambda=0.9 --learning_rate=3e-4 \
    --control_mode="pd_joint_delta_pos" \
    --save_model --capture_video --track

CKPT_L0="runs/${EXP_L0}/final_ckpt.pt"
echo "Level 0 완료 → ${STUDY_DIR}/${CKPT_L0}"

# ──────────────────────────────────────────────
# Level 1: 랜덤 크기 큐브 (Level 0 체크포인트에서 시작)
# ──────────────────────────────────────────────
EXP_L1="allegro_curriculum_level1_${TAG}"
echo ""
echo "=== Level 1 학습 시작 (from Level 0 checkpoint) ==="
python "${PATCH}" "${PPO_SCRIPT}" \
    --env_id="AllegroRotateLevel1-v1" \
    --exp_name="${EXP_L1}" \
    --checkpoint="${CKPT_L0}" \
    --num_envs=${NUM_ENVS} --num_steps=50 \
    --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=100000000 --eval_freq=25 \
    --gamma=0.8 --gae_lambda=0.9 --learning_rate=1e-4 \
    --control_mode="pd_joint_delta_pos" \
    --save_model --capture_video --track

CKPT_L1="runs/${EXP_L1}/final_ckpt.pt"
echo "Level 1 완료 → ${STUDY_DIR}/${CKPT_L1}"

# ──────────────────────────────────────────────
# Level 2: YCB 물체, z축 회전 (Level 1 체크포인트에서 시작)
# ──────────────────────────────────────────────
EXP_L2="allegro_curriculum_level2_${TAG}"
echo ""
echo "=== Level 2 학습 시작 (from Level 1 checkpoint) ==="
python "${PATCH}" "${PPO_SCRIPT}" \
    --env_id="AllegroRotateLevel2-v1" \
    --exp_name="${EXP_L2}" \
    --checkpoint="${CKPT_L1}" \
    --num_envs=${NUM_ENVS} --num_steps=50 \
    --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=100000000 --eval_freq=25 \
    --gamma=0.8 --gae_lambda=0.9 --learning_rate=1e-4 \
    --control_mode="pd_joint_delta_pos" \
    --save_model --capture_video --track

CKPT_L2="runs/${EXP_L2}/final_ckpt.pt"
echo "Level 2 완료 → ${STUDY_DIR}/${CKPT_L2}"

# ──────────────────────────────────────────────
# Level 3: YCB 물체, 랜덤 축 회전 (Level 2 체크포인트에서 시작)
# ──────────────────────────────────────────────
EXP_L3="allegro_curriculum_level3_${TAG}"
echo ""
echo "=== Level 3 학습 시작 (from Level 2 checkpoint) ==="
python "${PATCH}" "${PPO_SCRIPT}" \
    --env_id="AllegroRotateLevel3-v1" \
    --exp_name="${EXP_L3}" \
    --checkpoint="${CKPT_L2}" \
    --num_envs=${NUM_ENVS} --num_steps=50 \
    --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=1500000000 --eval_freq=25 \
    --gamma=0.8 --gae_lambda=0.9 --learning_rate=5e-5 \
    --control_mode="pd_joint_delta_pos" \
    --save_model --capture_video --track

CKPT_L3="runs/${EXP_L3}/final_ckpt.pt"
echo ""
echo "=== 커리큘럼 학습 전체 완료 ==="
echo "  Level 0: ${STUDY_DIR}/${CKPT_L0}"
echo "  Level 1: ${STUDY_DIR}/${CKPT_L1}"
echo "  Level 2: ${STUDY_DIR}/${CKPT_L2}"
echo "  Level 3: ${STUDY_DIR}/${CKPT_L3}"
