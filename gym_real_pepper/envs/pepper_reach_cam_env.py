import qi
import gym
import vision_definitions
import time
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from PIL import Image
from .pepper_env import PepperEnv
from scipy.spatial.transform import Rotation


class PepperReachCamEnv(PepperEnv):
    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 ip,
                 port=9559,
                 sim_steps_per_action=10,
                 dense=False,
                 head_motion=True):
        self._dense = dense
        super(PepperReachCamEnv,
              self).__init__(ip=ip,
                             port=port,
                             sim_steps_per_action=sim_steps_per_action,
                             head_motion=head_motion)

    def close(self):
        self._video_service.unsubscribe(self._cam_bottom)

    def reset(self):
        self._reset_scene()

        return self._get_observation()

    def step(self, action):
        """
        Action in terms of desired joint positions. Last number is the speed of the movement.
        """
        self._perform_action(action)

        obs = self._get_observation()

        is_success = False
        is_safety_violated = False
        obj_pos = None
        is_looking_at_table = True

        info = {
            "is_success": is_success,
            "is_safety_violated": is_safety_violated,
            "object_position": obj_pos
        }
        reward = self._compute_reward(is_success, is_safety_violated, is_looking_at_table)
        done = is_success or is_safety_violated

        return (obs, reward, done, info)

    def _setup_scene(self):
        super(PepperReachCamEnv, self)._setup_scene()

        # Setup camera
        resolution = vision_definitions.kQQVGA
        colorSpace = vision_definitions.kRGBColorSpace
        fps = 20

        self._cam = self._video_service.subscribeCamera(
            "python_GVM", vision_definitions.kBottomCamera, resolution,
            colorSpace, fps)

    def _get_observation_space(self):
        obs = self._get_observation()

        return spaces.Dict(
            dict(
                camera=spaces.Box(
                    0,
                    255,
                    shape=obs["camera"].shape,
                    dtype=obs["camera"].dtype,
                ),
                camera_pose=spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=obs["camera_pose"].shape,
                    dtype=obs["camera_pose"].dtype,
                ),
                joints_state=spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=obs["joints_state"].shape,
                    dtype=obs["joints_state"].dtype,
                ),
            ))

    def _get_observation(self):
        img = self._get_image(self._cam)

        joint_p = self._motion_service.getAngles(self.CONTROLLABLE_JOINTS,
                                                 True)
        cam_pos_raw = self._motion_service.getPosition(
            "CameraBottom", 2, True)

        rot = Rotation.from_euler('xyz', cam_pos_raw[3:], degrees=False)
        rot_quat = rot.as_quat()

        result = {
            "camera":
            img,
            "camera_pose":
            np.concatenate([cam_pos_raw[0:3], rot_quat]).astype(np.float32),
            "joints_state":
            np.array(joint_p, dtype=np.float32)
        }

        return result

    def _compute_reward(self, is_success, is_safety_violated,
                        is_looking_at_table):
        if is_success:
            return 1.0

        if is_safety_violated:
            return -1.0

        if self._dense:
            if is_looking_at_table:
                return -0.001
            else:
                return -0.01

        return 0.0
