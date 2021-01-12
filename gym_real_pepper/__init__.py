from gym.envs.registration import register

register(
    id='PepperReachCam-v0',
    entry_point='gym_real_pepper.envs:PepperReachCamEnv',
)