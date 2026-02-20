"""병렬 환경을 한 화면에 시각화하는 스크립트.

parallel_in_single_scene=True 옵션으로 여러 환경의 물체를 한 씬에 배치합니다.
"""

import argparse

import gymnasium as gym

import mani_skill.envs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="RotateSingleObjectInHandLevel0-v1")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--spacing", type=float, default=1.0,
                        help="환경 간 간격 (미터). 기본 5m이 너무 넓어서 1m으로 줄임")
    args = parser.parse_args()

    env = gym.make(
        args.env_id,
        num_envs=args.num_envs,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode="human",
        parallel_in_single_scene=True,
        sim_config=dict(
            spacing=args.spacing,
            gpu_memory_config=dict(
                max_rigid_patch_count=2**18,
                max_rigid_contact_count=2**21,
                found_lost_pairs_capacity=2**26,
            ),
        ),
    )

    env.reset(seed=42)
    while True:
        action = env.action_space.sample()
        env.step(action)
        env.render()


if __name__ == "__main__":
    main()
