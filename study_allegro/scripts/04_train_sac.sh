#!/bin/bash
# Allegro Hand SAC 학습 스크립트
#
# SAC는 PPO보다 sample-efficient하지만 wall-clock은 더 느릴 수 있음.
# 더 적은 병렬 환경 + 큰 리플레이 버퍼 사용.
#
# 사용법:
#   bash study_allegro/scripts/04_train_sac.sh [level] [num_envs] [timesteps]
#
# wandb 로깅을 끄려면 WANDB=0 환경변수 설정:
#   WANDB=0 bash study_allegro/scripts/04_train_sac.sh 0 64 2000000

set -e

# 프로젝트 루트와 study_allegro 디렉토리 경로 설정
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STUDY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$STUDY_DIR/.." && pwd)"

LEVEL=${1:-0}
NUM_ENVS=${2:-64}
TOTAL_TIMESTEPS=${3:-2000000}
WANDB_ENABLED=${WANDB:-1}

ENV_ID="RotateSingleObjectInHandLevel${LEVEL}-v1"
EXP_NAME="allegro_sac_level${LEVEL}_$(date +%Y%m%d_%H%M%S)"

WANDB_ARGS=""
if [ "${WANDB_ENABLED}" = "1" ]; then
    WANDB_ARGS="--track --wandb_project_name=ManiSkill3_AllegroHand"
fi

echo "=== Allegro Hand SAC Training ==="
echo "  Env: ${ENV_ID}"
echo "  Num envs: ${NUM_ENVS}"
echo "  Total timesteps: ${TOTAL_TIMESTEPS}"
echo "  WandB: ${WANDB_ENABLED}"
echo "  Output dir: ${STUDY_DIR}/runs/${EXP_NAME}/"
echo ""

# study_allegro/ 하위에 runs/, wandb/ 저장
cd "${STUDY_DIR}"
export WANDB_DIR="${STUDY_DIR}"

# 원본 sac.py에 eval_metrics KeyError 버그가 있어 패치 래퍼 사용
python "${STUDY_DIR}/scripts/patch_sac.py" \
    --env_id="${ENV_ID}" \
    --exp_name="${EXP_NAME}" \
    --num_envs=${NUM_ENVS} \
    --num_eval_envs=8 \
    --utd=0.5 \
    --buffer_size=1000000 \
    --total_timesteps=${TOTAL_TIMESTEPS} \
    --eval_freq=50000 \
    --control_mode="pd_joint_delta_pos" \
    --save_model \
    ${WANDB_ARGS}

echo ""
echo "=== Training complete ==="
echo "Model saved in: ${STUDY_DIR}/runs/${EXP_NAME}/"
