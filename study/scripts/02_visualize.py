"""Allegro Hand 시각화 스크립트

GUI 뷰어 또는 비디오 저장으로 환경을 시각적으로 확인.
"""

import argparse

import gymnasium as gym
import numpy as np

import mani_skill.envs


def render_video(env_id, output_path="study_allegro/videos/allegro_random.mp4", steps=200):
    """랜덤 액션으로 비디오 저장"""
    from mani_skill.utils.wrappers.record import RecordEpisode

    env = gym.make(
        env_id,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
    )
    env = RecordEpisode(env, output_dir="study_allegro/videos", save_trajectory=False)

    obs, _ = env.reset(seed=42)
    for _ in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
    print(f"Video saved to study_allegro/videos/")


def open_viewer(env_id):
    """GUI 뷰어 열기 (디스플레이 필요)"""
    env = gym.make(
        env_id,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode="human",
    )
    obs, _ = env.reset(seed=42)
    while True:
        action = env.action_space.sample() * 0  # 정지 상태
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="RotateSingleObjectInHandLevel0-v1")
    parser.add_argument("--mode", choices=["video", "viewer"], default="video")
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    if args.mode == "video":
        render_video(args.env_id, steps=args.steps)
    else:
        open_viewer(args.env_id)
