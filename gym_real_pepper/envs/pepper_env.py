import qi
import gym
import vision_definitions
import time
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from PIL import Image


class PepperEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    CONTROLLABLE_JOINTS = [
        "HeadYaw",
        "HeadPitch",
        "HipRoll",
        "LShoulderPitch",
        "LShoulderRoll",
        "LElbowYaw",
        "LElbowRoll",
        "LWristYaw",
        "LHand",
    ]

    FEATURE_LIMITS = [
        (-2.0857, 2.0857),
        (-0.7068, 0.4451),
        (-0.5149, 0.5149),
        (-2.0857, 2.0857),
        (0.0087, 1.5620),
        (-2.0857, 2.0857),
        (-1.5620, -0.0087),
        (-1.8239, 1.8239),
        (0, 1),
        (0, 1),
    ]

    def __init__(self,
                 ip,
                 port=9559,
                 gui=False,
                 sim_steps_per_action=10,
                 head_motion=True):
        self._sim_steps = sim_steps_per_action
        self._gui = gui

        self._session = qi.Session()
        self._session.connect("tcp://" + ip + ":" + str(port))
        self._motion_service = self._session.service("ALMotion")
        self._posture_service = self._session.service("ALRobotPosture")
        self._video_service = self._session.service("ALVideoDevice")
        self._life_service = self._session.service("ALAutonomousLife")

        self._setup_scene()

        if not head_motion:
            self.CONTROLLABLE_JOINTS = self.CONTROLLABLE_JOINTS[3:]
            self.FEATURE_LIMITS = self.FEATURE_LIMITS[3:]

        self.action_space = spaces.Box(-1.0,
                                       1.0,
                                       shape=(len(self.CONTROLLABLE_JOINTS) +
                                              1, ),
                                       dtype="float32")

        self.observation_space = self._get_observation_space()

    def __del__(self):
        self.close()

    def reset(self):
        raise "Not implemented"

    def step(self, action):
        raise "Not implemented"

    def render(self, mode="human"):
        pass

    def close(self):
        pass
        # self._life_service.setState("enabled")

    def seed(self, seed=None):
        np.random.seed(seed or 0)

    def _perform_action(self, action):
        action = list(action)
        assert len(action) == len(self.action_space.high.tolist())

        rescaled = [
            self._rescale_feature(i, f) for (i, f) in enumerate(action)
        ]
        angles = rescaled[:-1]
        speed = rescaled[-1]
        self._motion_service.setAngles(self.CONTROLLABLE_JOINTS, angles,
                                       [speed] * len(angles))

        time.sleep(self._sim_steps * 1. / 240.)

    def _setup_scene(self):
        # self._life_service.setState("disabled")
        self._posture_service.goToPosture("StandInit", 0.5)
        self._motion_service.setStiffnesses("Body", 1.0)

        self._motion_service.setAngles(
            ["KneePitch", "HipPitch", "LShoulderPitch", "HeadPitch"],
            [0.33, -0.9, -0.6, 0.3], [0.5] * 4)

        time.sleep(3)

        self._joints_initial_pose = self._motion_service.getAngles(
            "Body", True)

    def _reset_scene(self):
        self._reset_joint_state()
        raw_input("Reset complete?")

    def _reset_joint_state(self):
        self._motion_service.setAngles("Body", self._joints_initial_pose, 0.5)

    def _get_observation_space(self):
        raise "Not implemented"

    def _get_observation(self):
        raise "Not implemented"

    def _rescale_feature(self, index, value):
        r = self.FEATURE_LIMITS[index]
        return (r[1] - r[0]) * (value + 1) / 2 + r[0]

    def _get_image(self, camera_id):
        img = self._video_service.getImageRemote(camera_id)
        imageWidth = img[0]
        imageHeight = img[1]
        array = img[6]
        image_string = str(bytearray(array))
        # Create a PIL Image from our pixel array.
        im = Image.frombytes("RGB", (imageWidth, imageHeight), image_string)

        return np.array(im)
