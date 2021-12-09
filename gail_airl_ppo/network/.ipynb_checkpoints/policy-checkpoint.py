import torch
from torch import nn
import torch.nn.functional as F
from .utils import build_mlp, build_cnn, reparameterize, evaluate_lop_pi
from torch.distributions import Categorical


class StateIndependentPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh()):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def forward(self, states):
        return torch.tanh(self.net(states))

    def sample(self, states):
        return reparameterize(self.net(states), self.log_stds)

    def evaluate_log_pi(self, states, actions):
        return evaluate_lop_pi(self.net(states), self.log_stds, actions)


class StateDependentPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(256, 256),
                 hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=2 * action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states):
        return torch.tanh(self.net(states).chunk(2, dim=-1)[0])

    def sample(self, states):
        means, log_stds = self.net(states).chunk(2, dim=-1)
        return reparameterize(means, log_stds.clamp_(-20, 2))

class CNNStateIndependentPolicy(nn.Module):

    def __init__(self, state_shape, num_actions, hidden_units=64,
                 hidden_activation=nn.Tanh()):
        super().__init__()

        self.net = build_cnn(
            input_channels=4,
            output_dim=num_actions,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states):
        return self.net(states)

    def sample(self, states):
        probs = F.softmax(self.forward(states))
        dist = Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action)

    def evaluate_log_pi(self, states, actions):
        probs = F.softmax(self.forward(states))
        dist = Categorical(probs)
        return dist.log_prob(actions)

