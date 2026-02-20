"""GPU 병렬 비주얼 데이터 수집 학습 스크립트.

ManiSkill3의 핵심 기능 중 하나: GPU에서 수백~수천 개 환경의 RGBD + Segmentation 데이터를
동시에 렌더링하여 30,000+ FPS로 수집할 수 있다.

이 스크립트에서 배우는 것:
  1. obs_mode 조합별 데이터 형태와 내용 (rgb, depth, segmentation, rgbd, ...)
  2. 병렬 환경 수에 따른 렌더링 FPS 벤치마크
  3. 카메라 해상도/개수 변경이 성능에 미치는 영향
  4. 수집한 비주얼 데이터를 이미지로 저장하는 방법

사용법:
  python study_allegro/scripts/08_gpu_visual_data.py                      # 전체 실행
  python study_allegro/scripts/08_gpu_visual_data.py --section explore     # obs_mode별 데이터 구조 탐색
  python study_allegro/scripts/08_gpu_visual_data.py --section benchmark   # FPS 벤치마크
  python study_allegro/scripts/08_gpu_visual_data.py --section save        # 샘플 이미지 저장
"""

import argparse
import os
import time

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs


def print_header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# ──────────────────────────────────────────────────────────────────────
# Section 1: obs_mode별 데이터 구조 탐색
# ──────────────────────────────────────────────────────────────────────
def explore_obs_modes():
    """다양한 obs_mode로 환경을 만들어 반환되는 데이터의 구조와 크기를 확인한다."""
    print_header("Section 1: obs_mode별 데이터 구조 탐색")

    # 테스트할 obs_mode 목록
    obs_modes = [
        "state",                      # 비주얼 없음, 상태 벡터만
        "rgb",                        # RGB 이미지만
        "depth",                      # 깊이 이미지만
        "rgbd",                       # RGB + Depth
        "segmentation",               # 세그멘테이션 맵만
        "rgb+depth+segmentation",     # 전체 비주얼
        "pointcloud",                 # 포인트클라우드 (RGB+위치+세그 → 3D)
    ]

    for obs_mode in obs_modes:
        print(f"\n--- obs_mode = \"{obs_mode}\" ---")
        try:
            env = gym.make(
                "PickCube-v1",
                num_envs=2,
                obs_mode=obs_mode,
                control_mode="pd_joint_delta_pos",
            )
            obs, _ = env.reset(seed=42)

            def describe(data, prefix="  "):
                """obs 딕셔너리를 재귀적으로 탐색하며 키, 형태, dtype을 출력."""
                if isinstance(data, dict):
                    for k, v in data.items():
                        describe(v, prefix=f"{prefix}{k}/")
                elif isinstance(data, torch.Tensor):
                    print(f"{prefix}: shape={list(data.shape)}, dtype={data.dtype}")
                elif isinstance(data, np.ndarray):
                    print(f"{prefix}: shape={list(data.shape)}, dtype={data.dtype}")
                else:
                    print(f"{prefix}: type={type(data)}")

            if isinstance(obs, dict):
                describe(obs)
            else:
                print(f"  obs: shape={list(obs.shape)}, dtype={obs.dtype}")

            env.close()
        except Exception as e:
            print(f"  [SKIP] {e}")

    print("\n핵심 포인트:")
    print("  - 'sensor_data' 키 아래에 카메라별 데이터가 들어있음")
    print("  - RGB: shape [num_envs, H, W, 3], uint8")
    print("  - Depth: shape [num_envs, H, W, 1], float32 (미터 단위)")
    print("  - Segmentation: shape [num_envs, H, W, 1], int32 (물체 ID)")
    print("  - Pointcloud: sensor_data 대신 pointcloud 키에 xyzw, rgb, segmentation")


# ──────────────────────────────────────────────────────────────────────
# Section 2: 병렬 환경 수에 따른 렌더링 FPS 벤치마크
# ──────────────────────────────────────────────────────────────────────
def benchmark_visual_fps():
    """num_envs를 늘려가며 비주얼 데이터 수집 FPS를 측정한다."""
    print_header("Section 2: GPU 비주얼 렌더링 FPS 벤치마크")

    env_counts = [16, 64, 256, 1024]
    obs_modes_to_test = ["state", "rgbd", "rgb+depth+segmentation"]
    n_steps = 200

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"측정 방법: {n_steps} step 실행 후 총 FPS (= num_envs × steps / time)")
    print()

    results = []

    for obs_mode in obs_modes_to_test:
        print(f"--- obs_mode = \"{obs_mode}\" ---")
        for num_envs in env_counts:
            try:
                env = gym.make(
                    "PickCube-v1",
                    num_envs=num_envs,
                    obs_mode=obs_mode,
                    control_mode="pd_joint_delta_pos",
                )
                env.reset(seed=42)

                # Warmup
                for _ in range(5):
                    env.step(env.action_space.sample())

                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(n_steps):
                    actions = torch.rand(env.action_space.shape, device="cuda") * 2 - 1
                    env.step(actions)
                torch.cuda.synchronize()
                dt = time.time() - t0

                fps = num_envs * n_steps / dt
                results.append((obs_mode, num_envs, fps))
                print(f"  num_envs={num_envs:>5d}  →  {fps:>10,.0f} FPS  ({dt:.2f}s)")
                env.close()
            except Exception as e:
                print(f"  num_envs={num_envs:>5d}  →  [ERROR] {e}")
        print()

    # 비교 테이블
    if results:
        print("--- 요약 ---")
        print(f"{'obs_mode':<30s} {'num_envs':>10s} {'FPS':>12s}")
        print("-" * 55)
        for obs_mode, num_envs, fps in results:
            print(f"{obs_mode:<30s} {num_envs:>10d} {fps:>12,.0f}")

    print("\n핵심 포인트:")
    print("  - state 모드가 가장 빠름 (렌더링 없음)")
    print("  - rgbd는 RGB+Depth 렌더링 추가, 그래도 수만 FPS 달성 가능")
    print("  - segmentation 추가 시 약간의 오버헤드")
    print("  - num_envs를 늘릴수록 GPU 활용률이 높아져 FPS 증가 (VRAM 한계까지)")


# ──────────────────────────────────────────────────────────────────────
# Section 3: 수집한 비주얼 데이터를 이미지로 저장
# ──────────────────────────────────────────────────────────────────────
def save_visual_samples():
    """4개 병렬 환경의 RGB, Depth, Segmentation 이미지를 저장한다."""
    print_header("Section 3: 비주얼 데이터 샘플 저장")

    try:
        from PIL import Image
    except ImportError:
        print("[ERROR] Pillow가 필요합니다: uv pip install Pillow")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_envs = 4
    env = gym.make(
        "PickCube-v1",
        num_envs=num_envs,
        obs_mode="rgb+depth+segmentation",
        control_mode="pd_joint_delta_pos",
    )

    obs, _ = env.reset(seed=42)
    # 몇 스텝 진행 (초기 상태보다 흥미로운 장면)
    for _ in range(30):
        obs, _, _, _, _ = env.step(env.action_space.sample())

    out_dir = "study_allegro/videos/visual_samples"
    os.makedirs(out_dir, exist_ok=True)

    # 카메라 이름 목록
    cam_names = list(obs["sensor_data"].keys())
    print(f"카메라 {len(cam_names)}개: {cam_names}")

    for cam_name in cam_names:
        cam_data = obs["sensor_data"][cam_name]

        fig, axes = plt.subplots(num_envs, 3, figsize=(12, 4 * num_envs))
        if num_envs == 1:
            axes = axes[np.newaxis, :]

        for env_i in range(num_envs):
            # RGB
            rgb = cam_data["rgb"][env_i].cpu().numpy()
            axes[env_i, 0].imshow(rgb)
            axes[env_i, 0].set_title(f"Env {env_i} - RGB")
            axes[env_i, 0].axis("off")

            # Depth
            depth = cam_data["depth"][env_i, :, :, 0].cpu().numpy()
            depth_vis = depth.copy()
            depth_vis[depth_vis == np.inf] = 0
            axes[env_i, 1].imshow(depth_vis, cmap="viridis")
            axes[env_i, 1].set_title(f"Env {env_i} - Depth")
            axes[env_i, 1].axis("off")

            # Segmentation
            seg = cam_data["segmentation"][env_i, :, :, 0].cpu().numpy()
            axes[env_i, 2].imshow(seg, cmap="tab20")
            axes[env_i, 2].set_title(f"Env {env_i} - Segmentation")
            axes[env_i, 2].axis("off")

        fig.suptitle(f"Camera: {cam_name}", fontsize=14)
        fig.tight_layout()
        save_path = os.path.join(out_dir, f"{cam_name}_samples.png")
        fig.savefig(save_path, dpi=100)
        plt.close(fig)
        print(f"  저장: {save_path}")

    # 데이터 상세 정보
    print("\n--- 데이터 상세 ---")
    for cam_name in cam_names:
        cam_data = obs["sensor_data"][cam_name]
        for modality, tensor in cam_data.items():
            print(f"  {cam_name}/{modality}: shape={list(tensor.shape)}, "
                  f"dtype={tensor.dtype}, "
                  f"range=[{tensor.float().min().item():.3f}, {tensor.float().max().item():.3f}]")

    env.close()
    print(f"\n모든 샘플이 {out_dir}/ 에 저장되었습니다.")


def main():
    parser = argparse.ArgumentParser(description="GPU 비주얼 데이터 수집 학습")
    parser.add_argument(
        "--section",
        choices=["explore", "benchmark", "save", "all"],
        default="all",
        help="실행할 섹션 (기본: all)",
    )
    args = parser.parse_args()

    with torch.inference_mode():
        if args.section in ("explore", "all"):
            explore_obs_modes()
        if args.section in ("benchmark", "all"):
            benchmark_visual_fps()
        if args.section in ("save", "all"):
            save_visual_samples()


if __name__ == "__main__":
    main()
