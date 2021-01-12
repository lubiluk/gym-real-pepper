import qi
import gym
import vision_definitions
import time
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from PIL import Image

DISTANCE_THRESHOLD = 0.04
CONTROLLABLE_JOINTS = [
    "HipRoll",
    "HeadYaw",
    "HeadPitch",
    "LShoulderPitch",
    "LShoulderRoll",
    "LElbowYaw",
    "LElbowRoll",
    "LWristYaw",
    "LHand",
]


class PepperReachCamEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self, ip, port=9559,
        sim_steps_per_action=10,
        max_motion_speed=0.3,
        dense=True,
        depth_camera=False,
        top_camera=False,
    ):
        self._sim_steps = sim_steps_per_action
        self._max_speeds = [max_motion_speed] * len(CONTROLLABLE_JOINTS)
        self._dense = dense
        self._ip = ip
        self._port = port
        self._depth_camera = depth_camera
        self._top_camera = top_camera

        self._session = qi.Session()
        self._session.connect("tcp://" + ip + ":" + str(port))
        self._motion_service = self._session.service("ALMotion")
        self._posture_service = self._session.service("ALRobotPosture")
        self._video_service = self._session.service("ALVideoDevice")
        self._life_service = self._session.service("ALAutonomousLife")

        self._setup_scene()

        obs = self._get_observation()

        self.action_space = spaces.Box(
            -2.0857, 2.0857, shape=(len(CONTROLLABLE_JOINTS),), dtype="float32"
        )

        obs_spaces = dict(
            camera_bottom=spaces.Box(
                0,
                255,
                shape=obs["camera_bottom"].shape,
                dtype=obs["camera_bottom"].dtype,
            ),
            joints_state=spaces.Box(
                -np.inf,
                np.inf,
                shape=obs["joints_state"].shape,
                dtype=obs["joints_state"].dtype,
            ),
        )

        if self._top_camera:
            obs_spaces["camera_top"] = (
                spaces.Box(
                    0,
                    255,
                    shape=obs["camera_top"].shape,
                    dtype=obs["camera_top"].dtype,
                ),
            )

        if self._depth_camera:
            obs_spaces["camera_depth"] = spaces.Box(
                0,
                65535,
                shape=obs["camera_depth"].shape,
                dtype=obs["camera_depth"].dtype,
            )

        self.observation_space = spaces.Dict(obs_spaces)

    def reset(self):
        self._reset_scene()

        return self._get_observation()

    def step(self, action):
        action = list(action)
        assert len(action) == len(self.action_space.high.tolist())

        self._motion_service.setAngles(CONTROLLABLE_JOINTS, action, self._max_speeds)

        time.sleep(self._sim_steps * 1./240.)

        obs = self._get_observation()

        is_success = False
        info = {
            "is_success": False,
        }
        reward = 0.0
        done = False

        return (obs, reward, done, info)
        
    def render(self, mode="human"):
        pass

    def close(self):
        if self._top_camera:
            self._video_service.unsubscribe(self._cam_top)
        self._video_service.unsubscribe(self._cam_bottom)
        if self._depth_camera:
            self._video_service.unsubscribe(self._cam_depth)

    def seed(self, seed=None):
        np.random.seed(seed or 0)

    def _setup_scene(self):
        self._life_service.setState("disabled")
        self._posture_service.goToPosture("StandInit", 0.5)
        self._motion_service.setStiffnesses("Body", 1.0)
        self._motion_service.setAngles(
            ["KneePitch", "HipPitch", "LShoulderPitch"], [
                0.33, -0.9, -0.6], [0.5] * 3
        )

        time.sleep(3)

        self._joints_initial_pose = self._motion_service.getAngles("Body", True)

        resolution = vision_definitions.kQVGA
        colorSpace = vision_definitions.kRGBColorSpace
        fps = 20
        if self._top_camera:
            self._cam_top = self.self._video_service.subscribeCamera(
                "python_GVM", vision_definitions.kTopCamera, resolution, colorSpace, fps)
        self._cam_bottom = self._video_service.subscribeCamera(
            "python_GVM", vision_definitions.kBottomCamera, resolution, colorSpace, fps)
        if self._depth_camera:
            self._cam_depth = self.self._video_service.subscribeCamera(
                "python_GVM", vision_definitions.kDepthCamera, resolution, colorSpace, fps)

    def _reset_scene(self):
        self._motion_service.setAngles("Body", self._joints_initial_pose, 0.5)
        raw_input("Reset complete?")

    def _get_observation(self):
        img_bottom = self._get_image(self._cam_bottom)
        joint_p = self._motion_service.getAngles(CONTROLLABLE_JOINTS, True)
        # joint_v = [0.0] * len(CONTROLLABLE_JOINTS)

        result = {
            "camera_bottom": img_bottom,
            "joints_state": np.array(joint_p).astype(np.float32),
        }

        if self._top_camera:
            img_top = self._video_service.getImageRemote(self._cam_top)
            result["camera_top"] = img_top

        if self._depth_camera:
            img_depth = self._video_service.getImageRemote(self._cam_depth)
            result["camera_depth"] = img_depth

        return result

    def _get_image(self, camera_id):
        img = self._video_service.getImageRemote(camera_id)
        imageWidth = img[0]
        imageHeight = img[1]
        array = img[6]
        image_string = str(bytearray(array))
        # Create a PIL Image from our pixel array.
        im = Image.frombytes("RGB", (imageWidth, imageHeight), image_string)

        return np.array(im)
