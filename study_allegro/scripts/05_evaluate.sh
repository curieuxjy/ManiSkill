#!/bin/bash
# 학습된 모델 평가 스크립트
#
# 사용법:
#   bash study_allegro/scripts/05_evaluate.sh <algorithm> <checkpoint_path> [level]
#
# 체크포인트 경로는 study_allegro/ 기준 상대경로 또는 절대경로:
#   bash study_allegro/scripts/05_evaluate.sh ppo runs/allegro_ppo_.../final_ckpt.pt 0
#   bash study_allegro/scripts/05_evaluate.sh sac runs/allegro_sac_.../ckpt_50000.pt 0

set -e

# 프로젝트 루트와 study_allegro 디렉토리 경로 설정
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STUDY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$STUDY_DIR/.." && pwd)"

ALGO=${1:?"Usage: $0 <ppo|sac> <checkpoint_path> [level]"}
CHECKPOINT=${2:?"Usage: $0 <ppo|sac> <checkpoint_path> [level]"}
LEVEL=${3:-0}

ENV_ID="RotateSingleObjectInHandLevel${LEVEL}-v1"

echo "=== Evaluating Allegro Hand (${ALGO}) ==="
echo "  Env: ${ENV_ID}"
echo "  Checkpoint: ${CHECKPOINT}"
echo ""

# study_allegro/ 하위에서 실행 (eval 비디오도 여기에 저장)
cd "${STUDY_DIR}"

if [ "${ALGO}" = "ppo" ]; then
    python "${PROJECT_ROOT}/examples/baselines/ppo/ppo.py" \
        --env_id="${ENV_ID}" \
        --evaluate \
        --checkpoint="${CHECKPOINT}" \
        --num_eval_envs=8 \
        --num_eval_steps=300 \
        --control_mode="pd_joint_delta_pos" \
        --capture_video

elif [ "${ALGO}" = "sac" ]; then
    # SAC 원본에 eval_metrics KeyError 버그가 있어 패치 래퍼 사용
    python "${STUDY_DIR}/scripts/patch_sac.py" \
        --env_id="${ENV_ID}" \
        --evaluate \
        --checkpoint="${CHECKPOINT}" \
        --num_eval_envs=8 \
        --num_eval_steps=300 \
        --control_mode="pd_joint_delta_pos" \
        --capture_video

else
    echo "Error: algorithm must be 'ppo' or 'sac', got '${ALGO}'"
    exit 1
fi
