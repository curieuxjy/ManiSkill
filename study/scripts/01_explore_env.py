"""Allegro Hand 환경 탐색 스크립트

환경 생성, 관찰 공간/행동 공간 확인, 랜덤 액션으로 시뮬레이션 실행.
"""

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs  # 환경 등록 트리거

def explore_env(env_id="RotateSingleObjectInHandLevel0-v1"):
    print(f"=== {env_id} ===\n")

    # 1. 환경 생성 (CPU, 단일 환경)
    env = gym.make(
        env_id,
        obs_mode="state_dict",  # 딕셔너리로 관찰값 구조 파악
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
    )

    # 2. 공간 정보 출력
    print("Action space:", env.action_space)
    print("Action shape:", env.action_space.shape)
    print()

    # 3. 초기 관찰값 구조 확인
    obs, info = env.reset(seed=42)
    print("Observation keys:")
    def print_obs(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                print(f"  {prefix}{k}/")
                print_obs(v, prefix=f"  {prefix}")
            elif isinstance(v, (np.ndarray, torch.Tensor)):
                print(f"  {prefix}{k}: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"  {prefix}{k}: {type(v).__name__} = {v}")
    print_obs(obs)
    print()

    # 4. 몇 스텝 실행
    print("Running 10 random steps...")
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rew_val = reward.item() if hasattr(reward, 'item') else reward
        term_val = terminated.item() if hasattr(terminated, 'item') else terminated
        succ = info.get('success', None)
        fail = info.get('fail', None)
        succ_val = succ.item() if hasattr(succ, 'item') else succ
        fail_val = fail.item() if hasattr(fail, 'item') else fail
        print(f"  step {step:2d}: reward={rew_val:.4f}, terminated={term_val}, "
              f"success={succ_val}, fail={fail_val}")

    env.close()
    print("\nDone!")


def explore_agent_details(env_id="RotateSingleObjectInHandLevel0-v1"):
    """에이전트(로봇)의 상세 정보 확인"""
    print(f"\n=== Agent Details ({env_id}) ===\n")

    env = gym.make(env_id, obs_mode="state")
    env.reset(seed=42)

    agent = env.unwrapped.agent

    print(f"Robot UID: {agent.uid}")
    print(f"DOF: {agent.robot.dof}")
    print(f"Joint names: {[j.name for j in agent.robot.active_joints]}")
    print(f"Tip link names: {agent.tip_link_names}")
    print(f"Palm link name: {agent.palm_link_name}")
    print()

    # 컨트롤러 정보
    print(f"Controller type: {type(agent.controller).__name__}")
    print(f"Controller config stiffness: {agent.controller.config.stiffness}")
    print(f"Controller config damping: {agent.controller.config.damping}")
    print()

    # 촉각 센서 (AllegroHandRightTouch)
    if hasattr(agent, "fsr_links"):
        print(f"FSR sensor count: {len(agent.fsr_links)}")
        print(f"Finger FSR links: {agent.finger_fsr_link_names}")
        print(f"Palm FSR links: {agent.palm_fsr_link_names}")

    env.close()


if __name__ == "__main__":
    explore_env()
    explore_agent_details()
