"""Allegro Hand 물체 회전 환경 — partial reset 버그 수정 버전.

원본 RotateSingleObjectInHand 환경을 상속하여 _initialize_actors의
obj_heights 텐서 크기 불일치 버그를 수정한다.

원본 버그:
  Level 1+에서 obj_heights shape이 [num_envs]인데, partial reset 시
  new_pos shape은 [len(env_idx)]이라 텐서 크기 불일치 RuntimeError 발생.

수정:
  obj_heights가 1개(Level 0, 공유 큐브)이면 브로드캐스트,
  num_envs개(Level 1+, 환경별 다른 물체)이면 env_idx로 슬라이싱.

등록된 환경:
  AllegroRotateLevel0-v1  (고정 큐브)
  AllegroRotateLevel1-v1  (랜덤 크기 큐브)
  AllegroRotateLevel2-v1  (YCB 물체, z축 회전)
  AllegroRotateLevel3-v1  (YCB 물체, 랜덤 축 회전)

사용법:
  이 모듈이 import되면 자동으로 환경이 등록된다.
  gym.make("AllegroRotateLevel0-v1", num_envs=512, ...)
"""

import torch
import torch.nn.functional as F

from mani_skill.envs.tasks.dexterity.rotate_single_object_in_hand import (
    RotateSingleObjectInHand,
)
from mani_skill.utils.registration import register_env


class AllegroRotate(RotateSingleObjectInHand):
    """RotateSingleObjectInHand의 partial reset 버그를 수정한 버전.

    _initialize_actors만 오버라이드하여 obj_heights 인덱싱을 수정한다.
    나머지 모든 기능(씬 로드, 보상, 관찰, 평가)은 원본 그대로 사용.
    """

    def _initialize_actors(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            b = len(env_idx)
            # Initialize object pose
            self.table_scene.initialize(env_idx)

            new_pos = torch.randn((b, 3)) * self.obj_init_pos_noise
            # --- 수정: partial reset 시 obj_heights 크기 맞추기 ---
            # Level 0: obj_heights shape [1] (공유 큐브) → 브로드캐스트
            # Level 1+: obj_heights shape [num_envs] → env_idx로 슬라이싱
            obj_h = (
                self.obj_heights
                if len(self.obj_heights) == 1
                else self.obj_heights[env_idx]
            )
            new_pos[:, 2] = (
                torch.abs(new_pos[:, 2]) + self.hand_init_height + obj_h
            )

            new_pose = torch.zeros((b, 7))
            new_pose[:, 0:3] = new_pos
            new_pose[:, 3:7] = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

            self.obj.set_pose(new_pose)

            # Initialize object axis
            if self.difficulty_level <= 2:
                axis = torch.ones((b,), dtype=torch.long) * 2
            else:
                axis = torch.randint(0, 3, (b,), dtype=torch.long)
            if not hasattr(self, "rot_dir"):
                self.rot_dir = F.one_hot(axis, num_classes=3)
            else:
                self.rot_dir[env_idx] = F.one_hot(axis, num_classes=3)

            # Sample a unit vector on the tangent plane of rotating axis
            vector_axis = (axis + 1) % 3
            vector = F.one_hot(vector_axis, num_classes=3).float()

            self.success_threshold = torch.pi * 4
            # Controller parameters
            stiffness = torch.tensor(self.agent.controller.config.stiffness)
            damping = torch.tensor(self.agent.controller.config.damping)
            force_limit = torch.tensor(self.agent.controller.config.force_limit)
            self.controller_param = (
                stiffness.expand(self.num_envs, self.agent.robot.dof[0]),
                damping.expand(self.num_envs, self.agent.robot.dof[0]),
                force_limit.expand(self.num_envs, self.agent.robot.dof[0]),
            )
            if not hasattr(self, "unit_vector"):
                self.unit_vector = vector
                self.prev_unit_vector = vector.clone()
                self.cum_rotation_angle = torch.zeros((b,))
            else:
                self.unit_vector[env_idx] = vector
                self.prev_unit_vector[env_idx] = vector.clone()
                self.cum_rotation_angle[env_idx] = 0.0


# ──────────────────────────────────────────────────────────────────────
# 환경 등록
# ──────────────────────────────────────────────────────────────────────

@register_env("AllegroRotateLevel0-v1", max_episode_steps=300)
class AllegroRotateLevel0(AllegroRotate):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            robot_init_qpos_noise=0.02,
            obj_init_pos_noise=0.02,
            difficulty_level=0,
            **kwargs,
        )


@register_env("AllegroRotateLevel1-v1", max_episode_steps=300)
class AllegroRotateLevel1(AllegroRotate):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            robot_init_qpos_noise=0.02,
            obj_init_pos_noise=0.02,
            difficulty_level=1,
            **kwargs,
        )


@register_env(
    "AllegroRotateLevel2-v1",
    max_episode_steps=300,
    asset_download_ids=["ycb"],
)
class AllegroRotateLevel2(AllegroRotate):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            robot_init_qpos_noise=0.02,
            obj_init_pos_noise=0.02,
            difficulty_level=2,
            **kwargs,
        )


@register_env(
    "AllegroRotateLevel3-v1",
    max_episode_steps=300,
    asset_download_ids=["ycb"],
)
class AllegroRotateLevel3(AllegroRotate):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            robot_init_qpos_noise=0.02,
            obj_init_pos_noise=0.02,
            difficulty_level=3,
            **kwargs,
        )
