"""GPU 병렬 이종(heterogeneous) 시뮬레이션 학습 스크립트.

ManiSkill3의 고급 기능: 각 병렬 환경이 완전히 다른 물체/씬을 가질 수 있다.
예) 환경 0에는 머그컵, 환경 1에는 바나나, 환경 2에는 스패너, ...

이 스크립트에서 배우는 것:
  1. set_scene_idxs(): 특정 sub-scene에만 물체를 배치하는 메커니즘
  2. Actor.merge(): 서로 다른 물체들을 하나의 인터페이스로 통합
  3. _batched_episode_rng: 환경별 독립적인 난수 생성
  4. reconfiguration_freq: 리셋 시 씬을 재구성하는 방법
  5. 이종 환경에서의 학습과 일반화

사용법:
  python study_allegro/scripts/10_gpu_heterogeneous.py                         # 전체
  python study_allegro/scripts/10_gpu_heterogeneous.py --section scene_idx     # set_scene_idxs 원리
  python study_allegro/scripts/10_gpu_heterogeneous.py --section merge         # Actor.merge 구조
  python study_allegro/scripts/10_gpu_heterogeneous.py --section hetero_env    # 이종 환경 실제 예시
  python study_allegro/scripts/10_gpu_heterogeneous.py --section reconfig      # reconfiguration_freq
"""

import argparse

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs.actor import Actor


def print_header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# ──────────────────────────────────────────────────────────────────────
# Section 1: set_scene_idxs 메커니즘
# ──────────────────────────────────────────────────────────────────────
def explore_scene_idxs():
    """set_scene_idxs()로 특정 sub-scene에만 물체를 배치하는 원리를 학습한다."""
    print_header("Section 1: set_scene_idxs — 환경별 물체 배치")

    print("""
    GPU 병렬 시뮬레이션에서 ManiSkill은 num_envs개의 sub-scene을 생성한다.
    기본적으로 build()는 모든 sub-scene에 같은 물체를 만들지만,
    set_scene_idxs([i])를 호출하면 sub-scene i에만 해당 물체가 생성된다.

    이것이 이종(heterogeneous) 시뮬레이션의 핵심 메커니즘이다.

    코드 패턴:
    ┌─────────────────────────────────────────────────────┐
    │ for i in range(num_envs):                           │
    │     builder = scene.create_actor_builder()           │
    │     builder.add_box_collision(half_size=[sizes[i]])  │  ← 환경마다 다른 크기
    │     builder.set_scene_idxs([i])                      │  ← 이 환경에만 배치
    │     objs.append(builder.build(name=f"obj-{i}"))      │
    │                                                     │
    │ merged = Actor.merge(objs, name="object")            │  ← 통합 인터페이스
    └─────────────────────────────────────────────────────┘
    """)

    # 실제로 이 패턴을 사용하는 환경: RotateSingleObjectInHandLevel1
    # Level 1은 각 환경마다 다른 크기의 큐브를 생성한다.
    num_envs = 4
    env = gym.make(
        "RotateSingleObjectInHandLevel1-v1",
        num_envs=num_envs,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
    )
    base_env: BaseEnv = env.unwrapped

    print(f"RotateSingleObjectInHandLevel1 (num_envs={num_envs}):")
    print(f"  각 환경마다 다른 크기의 큐브가 생성됨")
    print()

    # 물체의 scene_idxs 확인
    obj = base_env.obj
    print(f"  obj.name          = {obj.name}")
    print(f"  obj.merged        = {obj.merged}")
    print(f"  obj._scene_idxs   = {obj._scene_idxs.tolist()}")
    print(f"  len(obj._objs)    = {len(obj._objs)}  (= num_envs, 각각 다른 sub-scene)")
    print()

    # 각 환경의 큐브 크기가 다른지 확인
    print(f"  obj_heights (per env) = {base_env.obj_heights.tolist()}")
    print(f"  → 각 환경마다 다른 높이 = 다른 크기의 큐브!")

    # 통합 인터페이스로 모든 환경의 포즈를 한 번에 접근
    pose = obj.pose
    print(f"\n  obj.pose (통합 접근):")
    print(f"    position shape = {list(pose.p.shape)}  → [num_envs, 3]")
    for i in range(num_envs):
        print(f"    env {i}: pos = [{pose.p[i, 0]:.3f}, {pose.p[i, 1]:.3f}, {pose.p[i, 2]:.3f}]")

    env.close()

    print("\n핵심 포인트:")
    print("  - set_scene_idxs([i]): 물체를 sub-scene i에만 배치")
    print("  - 각 환경에서 다른 물체를 만든 뒤 Actor.merge()로 통합")
    print("  - merged Actor는 obj.pose, obj.set_pose() 등 배치 연산 지원")


# ──────────────────────────────────────────────────────────────────────
# Section 2: Actor.merge() 구조
# ──────────────────────────────────────────────────────────────────────
def explore_actor_merge():
    """Actor.merge()가 어떻게 이종 물체들을 하나의 인터페이스로 통합하는지 학습한다."""
    print_header("Section 2: Actor.merge — 이종 물체 통합 인터페이스")

    print("""
    Actor.merge()의 역할:
    ┌──────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │  Sub-scene 0: [머그컵]   ──┐                                     │
    │  Sub-scene 1: [바나나]   ──┤                                     │
    │  Sub-scene 2: [스패너]   ──┼──→  Actor.merge()  ──→  merged_obj  │
    │  Sub-scene 3: [캔]       ──┘                                     │
    │                                                                  │
    │  merged_obj.pose       → [4, 7]  (모든 환경의 포즈)              │
    │  merged_obj.set_pose() → 한 번에 모든 환경의 포즈 설정           │
    │  merged_obj.get_state()→ [4, 13] (pos, quat, vel)               │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘
    """)

    # YCB 물체를 사용하는 Level 2 환경으로 실제 확인
    num_envs = 4
    env = gym.make(
        "RotateSingleObjectInHandLevel2-v1",
        num_envs=num_envs,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
    )
    base_env: BaseEnv = env.unwrapped

    obj = base_env.obj
    print(f"YCB 이종 환경 (Level 2, num_envs={num_envs}):")
    print(f"  obj.name        = {obj.name}")
    print(f"  obj.merged      = {obj.merged}")
    print(f"  obj._scene_idxs = {obj._scene_idxs.tolist()}")
    print()

    # 각 환경에 어떤 물체가 들어있는지 확인
    print("  각 환경의 물체:")
    for i, sub_obj in enumerate(base_env._objs):
        print(f"    env {i}: {sub_obj.name}")

    # 통합 연산 시연
    print(f"\n  통합 연산:")
    state = obj.get_state()
    print(f"    obj.get_state() shape = {list(state.shape)}  → [num_envs, 13]")
    print(f"    → 형상이 다른 물체들이지만 같은 텐서로 관리됨!")

    pose = obj.pose
    print(f"    obj.pose.p shape = {list(pose.p.shape)}")
    print(f"    obj.pose.q shape = {list(pose.q.shape)}")

    vel = obj.get_linear_velocity()
    print(f"    obj.get_linear_velocity() shape = {list(vel.shape)}")

    env.close()

    print("\n핵심 포인트:")
    print("  - Actor.merge()는 물체가 달라도 통일된 배치 인터페이스 제공")
    print("  - 내부적으로 _scene_idxs로 어떤 물체가 어떤 환경에 있는지 추적")
    print("  - pose, velocity, state 등 모든 속성이 [num_envs, ...] 텐서")
    print("  - 이 덕분에 이종 시뮬레이션에서도 벡터화된 RL 학습이 가능")


# ──────────────────────────────────────────────────────────────────────
# Section 3: 이종 환경 실제 예시 (YCB + Allegro)
# ──────────────────────────────────────────────────────────────────────
def explore_heterogeneous_env():
    """실제 이종 환경이 어떻게 동작하는지 YCB + Allegro Hand로 시연한다."""
    print_header("Section 3: 이종 환경 실제 동작 — YCB × Allegro Hand")

    num_envs = 8
    env = gym.make(
        "RotateSingleObjectInHandLevel2-v1",
        num_envs=num_envs,
        obs_mode="state_dict",
        control_mode="pd_joint_delta_pos",
    )
    base_env: BaseEnv = env.unwrapped

    obs, _ = env.reset(seed=42)

    # 각 환경의 물체 목록
    print(f"num_envs = {num_envs}")
    print(f"\n각 환경의 YCB 물체:")
    for i, sub_obj in enumerate(base_env._objs):
        print(f"  env {i}: {sub_obj.name}")

    # 물체별 높이 (형상에 따라 다름)
    print(f"\n물체 높이 (bounding box 기반):")
    for i, h in enumerate(base_env.obj_heights.tolist()):
        print(f"  env {i}: {h:.4f} m  ({base_env._objs[i].name})")

    # 관찰 구조 확인 — 모든 환경이 같은 obs 구조를 공유
    print(f"\n관찰 (obs) 구조:")
    if isinstance(obs, dict):
        for key, val in obs.items():
            if isinstance(val, torch.Tensor):
                print(f"  {key}: shape={list(val.shape)}")
            elif isinstance(val, dict):
                for k2, v2 in val.items():
                    if isinstance(v2, torch.Tensor):
                        print(f"  {key}/{k2}: shape={list(v2.shape)}")

    print(f"\n→ 물체가 달라도 obs/action space 차원은 동일!")
    print(f"  이것이 이종 환경에서 단일 정책으로 학습할 수 있는 이유.")

    # 랜덤 액션으로 몇 스텝 진행
    print(f"\n--- 5스텝 랜덤 실행 ---")
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  step {step}: reward range=[{reward.min().item():.3f}, {reward.max().item():.3f}], "
              f"any_done={(terminated | truncated).any().item()}")

    env.close()

    print("\n핵심 포인트:")
    print("  - 각 환경에 다른 YCB 물체가 있지만 동일한 obs/action space")
    print("  - 정책 네트워크는 물체의 형상을 몰라도 촉각/위치 정보만으로 학습")
    print("  - 이종 환경 → 더 강건하고 일반화된 정책 학습 가능")
    print("  - ManiSkill3에서는 이것을 GPU에서 병렬로 돌릴 수 있음!")


# ──────────────────────────────────────────────────────────────────────
# Section 4: reconfiguration_freq — 리셋 시 씬 재구성
# ──────────────────────────────────────────────────────────────────────
def explore_reconfiguration():
    """reconfiguration_freq를 이용해 리셋마다 씬을 재구성하는 방법을 학습한다."""
    print_header("Section 4: reconfiguration_freq — 리셋 시 씬 재구성")

    print("""
    reconfiguration_freq의 동작:
    ┌──────────────────────────────────────────────────────────────────┐
    │  reconfiguration_freq = 0  (기본값)                              │
    │    → 환경 생성 시 한 번만 씬 구성, 이후 reset()에서 포즈만 변경  │
    │    → 빠름! 하지만 매번 같은 물체                                 │
    │                                                                  │
    │  reconfiguration_freq = 1                                        │
    │    → 매 reset()마다 _load_scene() 재호출                         │
    │    → 매번 다른 물체 로드 가능, 하지만 느림                       │
    │    → GPU 병렬 시뮬레이션과 함께 사용 불가 (num_envs=1만 가능)    │
    │                                                                  │
    │  reconfiguration_freq = N                                        │
    │    → N번 reset할 때마다 한 번 씬 재구성                          │
    └──────────────────────────────────────────────────────────────────┘

    GPU 병렬 시뮬레이션에서 이종 씬을 구현하려면:
      reconfiguration_freq = 0 (재구성 없음) +
      _load_scene()에서 set_scene_idxs([i])로 환경별 다른 물체 배치

    즉, 초기 구성 시 모든 다양성을 한 번에 만들어 놓는 것!
    """)

    # reconfiguration_freq=0 (기본, GPU 호환)
    print("--- reconfiguration_freq=0 (기본, GPU 호환) ---")
    env = gym.make(
        "RotateSingleObjectInHandLevel2-v1",
        num_envs=4,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        reconfiguration_freq=0,
    )
    base_env: BaseEnv = env.unwrapped

    env.reset(seed=42)
    objs_first = [o.name for o in base_env._objs]
    print(f"  첫 번째 reset 물체: {objs_first}")

    env.reset(seed=99)
    objs_second = [o.name for o in base_env._objs]
    print(f"  두 번째 reset 물체: {objs_second}")
    print(f"  → 같은 물체! (씬 재구성 없음, 포즈만 변경)")
    env.close()

    # reconfiguration_freq=1 (CPU, 매번 다른 물체)
    print("\n--- reconfiguration_freq=1 (CPU, 매번 다른 씬) ---")
    env = gym.make(
        "PickSingleYCB-v1",
        num_envs=1,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        reconfiguration_freq=1,
    )
    base_env: BaseEnv = env.unwrapped

    env.reset(seed=42)
    obj_first = base_env.obj.name
    print(f"  첫 번째 reset 물체: {obj_first}")

    env.reset(seed=99)
    obj_second = base_env.obj.name
    print(f"  두 번째 reset 물체: {obj_second}")
    print(f"  → 다른 물체 가능! (매 reset마다 씬 재구성)")
    print(f"  → 단, num_envs=1 (CPU)에서만 사용 가능")
    env.close()

    print("\n핵심 포인트:")
    print("  - GPU 병렬 환경: reconfiguration_freq=0 + set_scene_idxs로 초기에 다양성 확보")
    print("  - CPU 단일 환경: reconfiguration_freq=1로 매 에피소드 다른 물체 사용 가능")
    print("  - GPU 이종 환경은 '초기 구성 시 모든 다양성 배치' 패턴 사용")
    print("  - 이 방식 덕분에 GPU 병렬성을 유지하면서도 다양한 물체로 학습 가능")


def main():
    parser = argparse.ArgumentParser(description="GPU 이종 시뮬레이션 학습")
    parser.add_argument(
        "--section",
        choices=["scene_idx", "merge", "hetero_env", "reconfig", "all"],
        default="all",
        help="실행할 섹션 (기본: all)",
    )
    args = parser.parse_args()

    with torch.inference_mode():
        if args.section in ("scene_idx", "all"):
            explore_scene_idxs()
        if args.section in ("merge", "all"):
            explore_actor_merge()
        if args.section in ("hetero_env", "all"):
            explore_heterogeneous_env()
        if args.section in ("reconfig", "all"):
            explore_reconfiguration()


if __name__ == "__main__":
    main()
