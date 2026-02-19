"""SAC 스크립트의 eval 출력 버그를 런타임에 패치한 뒤 실행하는 래퍼.

원본: examples/baselines/sac/sac.py 411행에서
  eval_metrics_mean['success_once'] 키가 없으면 KeyError 발생.

이 래퍼는 원본 파일을 수정하지 않고, 패치된 복사본을 실행합니다.

사용법 (원본 sac.py와 동일한 인자):
  python study_allegro/scripts/patch_sac.py --env_id=... --num_envs=64 ...
"""

import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SAC_PATH = REPO_ROOT / "examples" / "baselines" / "sac" / "sac.py"

original = SAC_PATH.read_text()

patched = original.replace(
    'pbar.set_description(\n'
    '                f"success_once: {eval_metrics_mean[\'success_once\']:.2f}, "\n'
    '                f"return: {eval_metrics_mean[\'return\']:.2f}"\n'
    '            )',
    'pbar.set_description(\n'
    '                f"success_once: {eval_metrics_mean.get(\'success_once\', 0):.2f}, "\n'
    '                f"return: {eval_metrics_mean.get(\'return\', 0):.2f}"\n'
    '            )',
)

if patched == original:
    # 패턴 매칭 실패 시 단순 문자열 치환 시도
    patched = original.replace(
        "eval_metrics_mean['success_once']",
        "eval_metrics_mean.get('success_once', 0)",
    ).replace(
        "eval_metrics_mean['return']",
        "eval_metrics_mean.get('return', 0)",
    )

with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
    f.write(patched)
    tmp_path = f.name

sys.exit(subprocess.call([sys.executable, tmp_path] + sys.argv[1:]))
