"""GPU 병렬 시뮬레이션 & 상태 기반 데이터 수집 학습 스크립트.

ManiSkill3의 핵심 기능: PhysX GPU 백엔드로 수백~수천 개의 환경을 동시에 시뮬레이션하고,
모든 환경의 상태(state)를 배치 텐서로 한 번에 가져오거나 설정할 수 있다.

이 스크립트에서 배우는 것:
  1. CPU vs GPU 시뮬레이션 백엔드 차이와 자동 선택 메커니즘
  2. 배치 상태 수집: get_state_dict() / get_state()로 전체 환경 상태를 텐서로 가져오기
  3. 상태 저장/복원: set_state_dict()로 정확히 이전 상태로 되돌리기
  4. 병렬 환경 수에 따른 시뮬레이션 FPS 벤치마크
  5. 상태 기반 데이터를 이용한 대량 시뮬레이션 데이터 수집 패턴

사용법:
  python study_allegro/scripts/09_gpu_state_collection.py                      # 전체 실행
  python study_allegro/scripts/09_gpu_state_collection.py --section backend    # CPU vs GPU 비교
  python study_allegro/scripts/09_gpu_state_collection.py --section state      # 상태 수집/복원
  python study_allegro/scripts/09_gpu_state_collection.py --section benchmark  # FPS 벤치마크
  python study_allegro/scripts/09_gpu_state_collection.py --section collect    # 대량 데이터 수집
"""

import argparse
import time

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs
from mani_skill.envs.sapien_env import BaseEnv


def print_header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# ──────────────────────────────────────────────────────────────────────
# Section 1: CPU vs GPU 시뮬레이션 백엔드
# ──────────────────────────────────────────────────────────────────────
def explore_backends():
    """num_envs에 따라 자동으로 CPU/GPU 백엔드가 선택되는 메커니즘을 확인한다."""
    print_header("Section 1: CPU vs GPU 시뮬레이션 백엔드")

    configs = [
        (1, "auto"),      # num_envs=1 → CPU
        (2, "auto"),      # num_envs>1 → GPU
        (16, "auto"),     # GPU
    ]

    for num_envs, sim_backend in configs:
        env = gym.make(
            "PickCube-v1",
            num_envs=num_envs,
            obs_mode="state",
            control_mode="pd_joint_delta_pos",
            sim_backend=sim_backend,
        )
        base_env: BaseEnv = env.unwrapped

        print(f"num_envs={num_envs}, sim_backend='{sim_backend}':")
        print(f"  gpu_sim_enabled = {base_env.gpu_sim_enabled}")
        print(f"  device          = {base_env.device}")
        print(f"  num sub-scenes  = {len(base_env.scene.sub_scenes)}")
        print(f"  PhysX backend   = {type(base_env.scene.px).__name__}")

        obs, _ = env.reset(seed=42)
        if isinstance(obs, dict):
            sample_key = list(obs.keys())[0]
            sample_val = obs[sample_key]
            if isinstance(sample_val, torch.Tensor):
                print(f"  obs['{sample_key}'] device = {sample_val.device}")
        elif isinstance(obs, torch.Tensor):
            print(f"  obs device = {obs.device}")

        env.close()
        print()

    print("핵심 포인트:")
    print("  - num_envs=1 → PhysxCpuSystem (CPU 시뮬레이션)")
    print("  - num_envs>1 → PhysxGpuSystem (GPU 시뮬레이션, CUDA)")
    print("  - sim_backend='auto'가 기본값, num_envs에 따라 자동 선택")
    print("  - GPU 시뮬레이션에서는 모든 텐서가 CUDA 디바이스에 있음")
    print("  - 각 병렬 환경은 별도의 sub-scene으로 관리됨")


# ──────────────────────────────────────────────────────────────────────
# Section 2: 배치 상태 수집 & 복원
# ──────────────────────────────────────────────────────────────────────
def explore_state_collection():
    """get_state_dict()와 set_state_dict()로 상태를 저장하고 복원하는 방법을 학습한다."""
    print_header("Section 2: 배치 상태 수집 & 복원")

    num_envs = 8
    env = gym.make(
        "PickCube-v1",
        num_envs=num_envs,
        obs_mode="state_dict",
        control_mode="pd_joint_delta_pos",
    )
    base_env: BaseEnv = env.unwrapped

    obs_before, _ = env.reset(seed=42)

    # ── 2a. get_state_dict(): 구조화된 상태 ──
    print("--- 2a. get_state_dict() ---")
    state_dict = base_env.get_state_dict()

    print("state_dict 키 구조:")
    for top_key in state_dict:
        if isinstance(state_dict[top_key], dict):
            for name, tensor in state_dict[top_key].items():
                print(f"  {top_key}/{name}: shape={list(tensor.shape)}, dtype={tensor.dtype}")
        else:
            val = state_dict[top_key]
            if isinstance(val, torch.Tensor):
                print(f"  {top_key}: shape={list(val.shape)}")

    print()
    print("상태 텐서 의미:")
    print("  actors/xxx:     [num_envs, 13] = [pos(3), quat(4), lin_vel(3), ang_vel(3)]")
    print("  articulations/xxx: [num_envs, 7+2*dof] = [root_pose(7), root_vel(6), qpos(dof), qvel(dof)]")

    # ── 2b. get_state(): 플랫 벡터 ──
    print("\n--- 2b. get_state() (flatten) ---")
    flat_state = base_env.get_state()
    print(f"flat_state: shape={list(flat_state.shape)}, dtype={flat_state.dtype}")
    print(f"  → [num_envs, total_state_dim] 형태의 단일 텐서")

    # ── 2c. 상태 변경 후 복원 ──
    print("\n--- 2c. 상태 저장 → 변경 → 복원 ---")

    # 현재 큐브 위치 기록
    cube_pos_before = state_dict["actors"]["cube"][:, :3].clone()
    print(f"복원 전 큐브 위치 (env 0): {cube_pos_before[0].tolist()}")

    # 50스텝 진행 (큐브 위치가 바뀜)
    for _ in range(50):
        env.step(env.action_space.sample())

    state_after = base_env.get_state_dict()
    cube_pos_after = state_after["actors"]["cube"][:, :3].clone()
    print(f"50스텝 후 큐브 위치 (env 0): {cube_pos_after[0].tolist()}")

    moved = torch.norm(cube_pos_after - cube_pos_before, dim=-1)
    print(f"이동 거리: {moved.tolist()}")

    # 저장했던 상태로 복원
    base_env.set_state_dict(state_dict)
    state_restored = base_env.get_state_dict()
    cube_pos_restored = state_restored["actors"]["cube"][:, :3]
    print(f"복원 후 큐브 위치 (env 0): {cube_pos_restored[0].tolist()}")

    diff = torch.norm(cube_pos_restored - cube_pos_before, dim=-1)
    print(f"복원 오차: {diff.tolist()}  (0에 가까워야 함)")

    env.close()

    print("\n핵심 포인트:")
    print("  - get_state_dict(): 구조화된 딕셔너리로 상태 수집 (actors, articulations)")
    print("  - get_state(): 단일 플랫 텐서로 변환 (ML 입력용)")
    print("  - set_state_dict(): 저장된 상태로 정확히 복원 가능")
    print("  - 모든 환경의 상태가 배치 텐서 [num_envs, ...] 형태")
    print("  - GPU에서 실행되므로 CPU ↔ GPU 복사 없이 빠르게 처리")


# ──────────────────────────────────────────────────────────────────────
# Section 3: FPS 벤치마크 (state-based)
# ──────────────────────────────────────────────────────────────────────
def benchmark_sim_fps():
    """num_envs를 늘려가며 state-based 시뮬레이션 FPS를 측정한다."""
    print_header("Section 3: GPU 시뮬레이션 FPS 벤치마크 (state-based)")

    env_counts = [16, 64, 256, 1024, 4096]
    n_steps = 500

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"환경: PickCube-v1, obs_mode=state")
    print(f"측정: {n_steps} step (warmup 10 step 제외)")
    print()

    results = []
    for num_envs in env_counts:
        try:
            env = gym.make(
                "PickCube-v1",
                num_envs=num_envs,
                obs_mode="state",
                control_mode="pd_joint_delta_pos",
            )
            env.reset(seed=42)

            # Warmup
            for _ in range(10):
                env.step(env.action_space.sample())

            # step만 측정
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n_steps):
                actions = torch.rand(env.action_space.shape, device="cuda") * 2 - 1
                obs, rew, terminated, truncated, info = env.step(actions)
            torch.cuda.synchronize()
            dt_step = time.time() - t0
            fps_step = num_envs * n_steps / dt_step

            # step + 중간중간 reset
            env.reset(seed=42)
            torch.cuda.synchronize()
            t0 = time.time()
            for i in range(n_steps):
                actions = torch.rand(env.action_space.shape, device="cuda") * 2 - 1
                obs, rew, terminated, truncated, info = env.step(actions)
                if i % 100 == 0 and i != 0:
                    env.reset()
            torch.cuda.synchronize()
            dt_reset = time.time() - t0
            fps_reset = num_envs * n_steps / dt_reset

            results.append((num_envs, fps_step, fps_reset))
            print(f"  num_envs={num_envs:>5d}  "
                  f"step: {fps_step:>10,.0f} FPS  "
                  f"step+reset: {fps_reset:>10,.0f} FPS")
            env.close()
        except Exception as e:
            print(f"  num_envs={num_envs:>5d}  [ERROR] {e}")

    print("\n핵심 포인트:")
    print("  - state-based 시뮬레이션은 렌더링이 없어 가장 빠름")
    print("  - num_envs가 늘어날수록 GPU 활용률 증가 → 단위당 throughput 향상")
    print("  - reset 포함 시 약간 느려지지만, partial reset 덕분에 오버헤드 최소화")


# ──────────────────────────────────────────────────────────────────────
# Section 4: 대량 데이터 수집 패턴
# ──────────────────────────────────────────────────────────────────────
def collect_data_demo():
    """GPU 병렬 환경에서 대량의 transition 데이터를 수집하는 패턴을 시연한다."""
    print_header("Section 4: 대량 상태 기반 데이터 수집 패턴")

    num_envs = 64
    collect_steps = 200
    env = gym.make(
        "PickCube-v1",
        num_envs=num_envs,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
    )
    base_env: BaseEnv = env.unwrapped

    obs, _ = env.reset(seed=42)

    # 수집할 데이터 버퍼 (GPU 텐서)
    obs_dim = obs.shape[-1] if isinstance(obs, torch.Tensor) else sum(
        v.shape[-1] for v in obs.values() if isinstance(v, torch.Tensor)
    )
    all_obs = torch.zeros((collect_steps, num_envs, obs_dim), device=base_env.device)
    all_actions = torch.zeros((collect_steps, num_envs, env.action_space.shape[-1]),
                              device=base_env.device)
    all_rewards = torch.zeros((collect_steps, num_envs), device=base_env.device)
    all_dones = torch.zeros((collect_steps, num_envs), dtype=torch.bool,
                            device=base_env.device)

    print(f"수집 설정: {num_envs} envs × {collect_steps} steps = {num_envs * collect_steps:,} transitions")
    print(f"obs_dim={obs_dim}, action_dim={env.action_space.shape[-1]}")
    print()

    # 데이터 수집
    torch.cuda.synchronize()
    t0 = time.time()

    for step in range(collect_steps):
        # 랜덤 액션 (실제로는 정책 네트워크 출력)
        action = torch.rand(env.action_space.shape, device=base_env.device) * 2 - 1

        if isinstance(obs, torch.Tensor):
            all_obs[step] = obs
        all_actions[step] = action

        obs, reward, terminated, truncated, info = env.step(action)
        all_rewards[step] = reward
        all_dones[step] = terminated | truncated

    torch.cuda.synchronize()
    dt = time.time() - t0

    total_transitions = num_envs * collect_steps
    fps = total_transitions / dt

    print(f"수집 완료: {total_transitions:,} transitions in {dt:.2f}s ({fps:,.0f} FPS)")
    print()

    # 수집된 데이터 통계
    print("--- 수집 데이터 통계 ---")
    print(f"  obs     buffer: shape={list(all_obs.shape)}, device={all_obs.device}")
    print(f"  actions buffer: shape={list(all_actions.shape)}")
    print(f"  rewards buffer: shape={list(all_rewards.shape)}")
    print(f"  dones   buffer: shape={list(all_dones.shape)}")
    print()
    print(f"  보상 평균: {all_rewards.mean().item():.4f}")
    print(f"  보상 최대: {all_rewards.max().item():.4f}")
    print(f"  완료 에피소드 수: {all_dones.sum().item()}")
    print()

    # GPU → CPU 전송 시간 측정
    torch.cuda.synchronize()
    t0 = time.time()
    cpu_obs = all_obs.cpu()
    cpu_actions = all_actions.cpu()
    cpu_rewards = all_rewards.cpu()
    cpu_dones = all_dones.cpu()
    dt_transfer = time.time() - t0

    total_bytes = (cpu_obs.nbytes + cpu_actions.nbytes +
                   cpu_rewards.nbytes + cpu_dones.nbytes)
    print(f"GPU → CPU 전송: {total_bytes / 1024**2:.1f} MB in {dt_transfer*1000:.1f}ms")

    env.close()

    print("\n핵심 포인트:")
    print("  - 모든 데이터가 GPU 텐서로 바로 수집됨 (CPU 복사 불필요)")
    print("  - 정책 네트워크도 GPU에 있으면 전체 파이프라인이 GPU에서 실행")
    print("  - CPU 전송은 로깅/저장할 때만 필요")
    print("  - 이 패턴이 PPO/SAC 등 on-policy/off-policy RL의 데이터 수집 루프")


def main():
    parser = argparse.ArgumentParser(description="GPU 시뮬레이션 & 상태 수집 학습")
    parser.add_argument(
        "--section",
        choices=["backend", "state", "benchmark", "collect", "all"],
        default="all",
        help="실행할 섹션 (기본: all)",
    )
    args = parser.parse_args()

    with torch.inference_mode():
        if args.section in ("backend", "all"):
            explore_backends()
        if args.section in ("state", "all"):
            explore_state_collection()
        if args.section in ("benchmark", "all"):
            benchmark_sim_fps()
        if args.section in ("collect", "all"):
            collect_data_demo()


if __name__ == "__main__":
    main()
