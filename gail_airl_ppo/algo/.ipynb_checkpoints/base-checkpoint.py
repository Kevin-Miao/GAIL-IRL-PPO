from abc import ABC, abstractmethod
import os
import numpy as np
import torch


class Algorithm(ABC):

    def __init__(self, state_shape, action_shape, device, seed, gamma):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.learning_steps = 0
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.gamma = gamma
        if isinstance(self.action_shape, int):
            self.categorical = True
        else:
            self.categorical = False

    def explore(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        state = torch.moveaxis(state, 3, 1)
        with torch.no_grad():
            action, log_pi = self.actor.sample(state)
        return action.cpu().numpy(), log_pi.item()

    def exploit(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        state = torch.moveaxis(state, 3, 1)
        with torch.no_grad():
            action, _ = self.actor.sample(state)
        return action.cpu().numpy()

    @abstractmethod
    def is_update(self, step):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
