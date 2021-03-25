# Python 2.7
import gym
import gym_real_pepper
import time
import numpy as np
# import cv2

start = time.time()
env = gym.make("PepperReachCam-v0", ip='192.168.2.101')
end = time.time()
print("=== Make === {}".format(end - start))

start = time.time()
env.reset()
end = time.time()
print("=== Reset === {}".format(end - start))

start = time.time()
for _ in range(100):
    env.step([1.0] * 9 + [0.3])
for _ in range(100):
    env.step([-1.0] * 9 + [0.3])
end = time.time()
print("=== Act1 === {}".format(end - start))

env.close()