from typing import Any
from gym.wrappers import AtariPreprocessing
from gym.spaces import Box
import numpy as np
import cv2


class AtariWrapper(AtariPreprocessing):
    """
    Transforms a 210x160 RGB image into a 84x84 grayscale one. Flattens it so it can be plugged into the DQN
    """
    def __init__(self, env):
        super().__init__(env, scale_obs=True)
        self.stack_size = 4
        self.stacked_frame = np.array([])
        self.resetting = False

        self.observation_space = Box(shape=(84, 84, 4), dtype=np.float32, low=0, high=1)


    def reset(self, **kwargs):
        self.resetting = True
        obs = super().reset() # WATCH OUT - this isn't supporting kwargs for AtariPreprocessing
        self.resetting = False

        self.stacked_frame = np.zeros((self.screen_size, self.screen_size, self.stack_size))
        for i in range(self.stack_size):
            self.stacked_frame[:, :, i] = obs

        return self.stacked_frame



    def _get_obs(self):
        if not self.resetting:
            for i in range(1, self.stack_size, - 1):
                self.stacked_frame[:, :, i - 1] = self.stacked_frame[:, :, i]
            self.stacked_frame[:, :, self.stack_size - 1] = super()._get_obs()

            return self.stacked_frame
        return super()._get_obs()