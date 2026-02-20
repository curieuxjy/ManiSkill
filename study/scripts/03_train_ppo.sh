#!/bin/bash
# Allegro Hand PPO 학습 스크립트
#
# 사용법:
#   bash study_allegro/scripts/03_train_ppo.sh [level] [num_envs] [timesteps]
#
# wandb 로깅을 끄려면 WANDB=0 환경변수 설정:
#   WANDB=0 bash study_allegro/scripts/03_train_ppo.sh 0 512 5000000
#
# 예시:
#   bash study_allegro/scripts/03_train_ppo.sh 0 512 5000000
#   bash study_allegro/scripts/03_train_ppo.sh 1 256 10000000

set -e

# 프로젝트 루트와 study_allegro 디렉토리 경로 설정
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STUDY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$STUDY_DIR/.." && pwd)"

LEVEL=${1:-0}
NUM_ENVS=${2:-256}
TOTAL_TIMESTEPS=${3:-5000000}
WANDB_ENABLED=${WANDB:-1}

ENV_ID="AllegroRotateLevel${LEVEL}-v1"
EXP_NAME="allegro_ppo_level${LEVEL}_$(date +%Y%m%d_%H%M%S)"

WANDB_ARGS=""
if [ "${WANDB_ENABLED}" = "1" ]; then
    WANDB_ARGS="--track --wandb_project_name=ManiSkill3_AllegroHand"
fi

echo "=== Allegro Hand PPO Training ==="
echo "  Env: ${ENV_ID}"
echo "  Num envs: ${NUM_ENVS}"
echo "  Total timesteps: ${TOTAL_TIMESTEPS}"
echo "  Experiment: ${EXP_NAME}"
echo "  WandB: ${WANDB_ENABLED}"
echo "  Output dir: ${STUDY_DIR}/runs/${EXP_NAME}/"
echo ""

# study_allegro/ 하위에 runs/, wandb/ 저장
cd "${STUDY_DIR}"
export WANDB_DIR="${STUDY_DIR}"

python "${STUDY_DIR}/scripts/run_with_patch.py" "${PROJECT_ROOT}/examples/baselines/ppo/ppo.py" \
    --env_id="${ENV_ID}" \
    --exp_name="${EXP_NAME}" \
    --num_envs=${NUM_ENVS} \
    --num_steps=50 \
    --update_epochs=8 \
    --num_minibatches=32 \
    --total_timesteps=${TOTAL_TIMESTEPS} \
    --eval_freq=25 \
    --num_eval_envs=8 \
    --num_eval_steps=300 \
    --gamma=0.8 \
    --gae_lambda=0.9 \
    --learning_rate=3e-4 \
    --control_mode="pd_joint_delta_pos" \
    --save_model \
    --capture_video \
    ${WANDB_ARGS}

echo ""
echo "=== Training complete ==="
echo "Model saved in: ${STUDY_DIR}/runs/${EXP_NAME}/"
