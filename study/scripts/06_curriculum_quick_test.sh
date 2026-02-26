#!/bin/bash
# Allegro Hand 커리큘럼 레벨업 빠른 검증 스크립트
#
# 06_curriculum_train.sh의 축소 버전으로, 각 레벨이 정상적으로
# 체크포인트를 넘겨받아 학습되는지 확인하는 용도입니다.
#
# 사용법:
#   bash study/scripts/06_curriculum_quick_test.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STUDY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$STUDY_DIR/.." && pwd)"

NUM_ENVS=64
TOTAL_STEPS=500000
TAG="quick_$(date +%Y%m%d_%H%M%S)"

cd "${STUDY_DIR}"
export WANDB_DIR="${STUDY_DIR}"

PATCH="${STUDY_DIR}/scripts/run_with_patch.py"
PPO_SCRIPT="${PROJECT_ROOT}/examples/baselines/ppo/ppo.py"

COMMON_ARGS=(
    --num_envs=${NUM_ENVS} --num_steps=50
    --update_epochs=4 --num_minibatches=8
    --total_timesteps=${TOTAL_STEPS} --eval_freq=10
    --gamma=0.8 --gae_lambda=0.9
    --control_mode="pd_joint_delta_pos"
    --save_model
)

# ── Level 0 ──
EXP_L0="quick_level0_${TAG}"
echo "=== Level 0 ==="
python "${PATCH}" "${PPO_SCRIPT}" \
    --env_id="AllegroRotateLevel0-v1" \
    --exp_name="${EXP_L0}" \
    --learning_rate=3e-4 \
    "${COMMON_ARGS[@]}"

CKPT_L0="runs/${EXP_L0}/final_ckpt.pt"
echo "Level 0 done → ${CKPT_L0}"

# ── Level 1 ──
EXP_L1="quick_level1_${TAG}"
echo ""
echo "=== Level 1 (from L0) ==="
python "${PATCH}" "${PPO_SCRIPT}" \
    --env_id="AllegroRotateLevel1-v1" \
    --exp_name="${EXP_L1}" \
    --checkpoint="${CKPT_L0}" \
    --learning_rate=1e-4 \
    "${COMMON_ARGS[@]}"

CKPT_L1="runs/${EXP_L1}/final_ckpt.pt"
echo "Level 1 done → ${CKPT_L1}"

# ── Level 2 ──
EXP_L2="quick_level2_${TAG}"
echo ""
echo "=== Level 2 (from L1) ==="
python "${PATCH}" "${PPO_SCRIPT}" \
    --env_id="AllegroRotateLevel2-v1" \
    --exp_name="${EXP_L2}" \
    --checkpoint="${CKPT_L1}" \
    --learning_rate=1e-4 \
    "${COMMON_ARGS[@]}"

CKPT_L2="runs/${EXP_L2}/final_ckpt.pt"
echo "Level 2 done → ${CKPT_L2}"

# ── Level 3 ──
EXP_L3="quick_level3_${TAG}"
echo ""
echo "=== Level 3 (from L2) ==="
python "${PATCH}" "${PPO_SCRIPT}" \
    --env_id="AllegroRotateLevel3-v1" \
    --exp_name="${EXP_L3}" \
    --checkpoint="${CKPT_L2}" \
    --learning_rate=5e-5 \
    "${COMMON_ARGS[@]}"

CKPT_L3="runs/${EXP_L3}/final_ckpt.pt"

echo ""
echo "=== 전체 레벨업 검증 완료 ==="
echo "  Level 0: ${STUDY_DIR}/${CKPT_L0}"
echo "  Level 1: ${STUDY_DIR}/${CKPT_L1}"
echo "  Level 2: ${STUDY_DIR}/${CKPT_L2}"
echo "  Level 3: ${STUDY_DIR}/${CKPT_L3}"
