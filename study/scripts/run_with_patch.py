"""커스텀 환경을 등록한 뒤 대상 스크립트를 실행하는 래퍼.

메인 소스를 수정하지 않고, study/envs/에 정의된 커스텀 환경을
gym에 등록한 뒤 원본 학습 스크립트를 실행한다.

등록되는 환경:
  AllegroRotateLevel0-v1 ~ AllegroRotateLevel3-v1
  (RotateSingleObjectInHand의 partial reset 버그 수정 버전)

추가 패치:
  - sac.py 대상일 경우 eval_metrics KeyError 수정

사용법:
  python study/scripts/run_with_patch.py <script.py> [args...]

예시:
  python study/scripts/run_with_patch.py examples/baselines/ppo/ppo.py \\
      --env_id=AllegroRotateLevel0-v1 --num_envs=512 ...
"""

import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
STUDY_DIR = SCRIPT_DIR.parent
REPO_ROOT = STUDY_DIR.parent
CUSTOM_ENV_PATH = STUDY_DIR / "envs" / "allegro_rotate.py"

target_script = Path(sys.argv[1]).resolve()
original = target_script.read_text()

# 1. 커스텀 환경 등록 코드를 주입 (import mani_skill.envs 직후)
env_register_inject = (
    f'\n# --- study custom envs ---\n'
    f'exec(open(r"{CUSTOM_ENV_PATH}").read())\n'
    f'# --- end custom envs ---\n'
)

if "import mani_skill.envs" in original:
    patched = original.replace(
        "import mani_skill.envs",
        f"import mani_skill.envs{env_register_inject}",
        1,
    )
else:
    patched = env_register_inject + original

# 2. SAC eval_metrics KeyError 패치 (sac.py인 경우)
if "sac" in target_script.name.lower():
    patched = patched.replace(
        "eval_metrics_mean['success_once']",
        "eval_metrics_mean.get('success_once', 0)",
    ).replace(
        "eval_metrics_mean['return']",
        "eval_metrics_mean.get('return', 0)",
    )

with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
    f.write(patched)
    tmp_path = f.name

sys.exit(subprocess.call([sys.executable, tmp_path] + sys.argv[2:]))
