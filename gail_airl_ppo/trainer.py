import os
from time import time, sleep
from datetime import timedelta
from torch.utils.tensorboard import SummaryWriter
from gym.wrappers import Monitor

# Video Stuff

import gym
import imageio
import numpy as np
import base64
import IPython
import PIL.Image
import pyvirtualdisplay
from pathlib import Path
from IPython import display as ipythondisplay

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import VecVideoRecorder, SubprocVecEnv, DummyVecEnv
from stable_baselines import PPO2

class Trainer:

    def __init__(self, env, env_test, algo, log_dir, seed=0, num_steps=10**5,
                 eval_interval=10**3, num_eval_episodes=5, video=False, env_id=None):
        super().__init__()

        # Env to collect samples.
        self.env = env
        self.env.seed(seed)

        # Env for evaluation.
        self.env_test = env_test
        self.env_test.seed(2**31-seed)
        self.env_id = env_id

        self.video = video
        if video:
            self.env_test_video = Monitor(env_test, './video', force=True)

        self.algo = algo
        self.log_dir = log_dir

        # Log setting.
        self.summary_dir = os.path.join(log_dir, 'summary')
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.model_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Other parameters.
        self.num_steps = num_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes

    def train(self):
        # Time to start training.
        self.start_time = time()
        # Episode's timestep.
        t = 0
        # Initialize the environment.
        state = self.env.reset()

        for step in range(1, self.num_steps + 1):
            # Pass to the algorithm to update state and episode timestep.
            state, t = self.algo.step(self.env, state, t, step)

            # Update the algorithm whenever ready.
            if self.algo.is_update(step):
                self.algo.update(self.writer)

            # Evaluate regularly.
            if step % self.eval_interval == 0:
                self.evaluate(step)
                self.algo.save_models(
                    os.path.join(self.model_dir, f'step{step}'))

        self.record_video()

        # Wait for the logging to be finished.
        sleep(10)

    def evaluate(self, step):
        mean_return = 0.0

        env = self.env_test
            
        for i in range(self.num_eval_episodes):
            state = env.reset()
            episode_return = 0.0
            done = False

            while (not done):
                action = self.algo.exploit(state)
                state, reward, done, _ = env.step(action)
                episode_return += reward

            mean_return += episode_return / self.num_eval_episodes

        self.writer.add_scalar('return/test', mean_return, step)
        print(f'Num steps: {step:<6}   '
              f'Return: {mean_return:<5.1f}   '
              f'Time: {self.time}')

    # Record video
    def record_video(self, video_length=500, prefix='', video_folder='videos/'):
        """
        :param video_length: (int)
        :param prefix: (str)
        :param video_folder: (str)
        """
        os.system("Xvfb :1 -screen 0 1024x768x24 &")
        os.environ['DISPLAY'] = ':1'
            
        if not os.path.exists(video_folder):
            os.mkdir(video_folder)

        env = DummyVecEnv([lambda: gym.make(env_id)])
        eval_env = DummyVecEnv([lambda: gym.make(self.env_id)])
        
        # Start the video at step=0 and record 500 steps
        eval_env = VecVideoRecorder(env, video_folder=video_folder,
                                    record_video_trigger=lambda step: step == 0, video_length=video_length,
                                    name_prefix=prefix)

        obs = eval_env.reset()
        for _ in range(video_length):
            action, _ = self.algo.exploit(obs)
            obs, _, _, _ = eval_env.step(action)

        # Close the video recorder
        eval_env.close()

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))
