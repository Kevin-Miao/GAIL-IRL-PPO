import math
import torch
from torch import nn

class AttCNN(nn.Module):
    def __init__(self, input_channels, output_dim, hidden_units=64, hidden_activation=nn.Tanh(), output_activation=None):
        super().__init__()
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.att_conv3  = nn.Conv2d(hidden_units, 1, kernel_size=3, padding=1,bias=False)
        self.bn_att3 = nn.BatchNorm2d(1)
        self.att_gap = nn.AvgPool2d(14)
        self.sigmoid = nn.Sigmoid()
        self.cnn = nn.Sequential(
                nn.Conv2d(self.input_channels, 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),)
        self.out = nn.Sequential(
                nn.Flatten(),
                nn.Linear(5184, self.hidden_units),
                self.hidden_activation,
                nn.Linear(self.hidden_units, self.output_dim)
            )

    def forward(self, x):
        x = self.cnn(x)
        self.att = self.sigmoid(self.bn_att3(self.att_conv3(x)))
        x = x + self.att * x
        if self.output_activation:
            return nn.Sequential(self.out, self.output_activation)(x), self.att
        else:
            return self.out(x), self.att

def build_mlp(input_dim, output_dim, hidden_units=[64, 64],
              hidden_activation=nn.Tanh(), output_activation=None):
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units
    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)

def build_cnn(input_channels, output_dim, hidden_units=64,
              hidden_activation=nn.Tanh(), output_activation=None):
    network = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(5184, hidden_units),
            hidden_activation,
            nn.Linear(hidden_units, output_dim)
        )
    if output_activation:
        return nn.Sequential(network, output_activation)
    return network

def build_att_cnn(input_channels, output_dim, hidden_units=64,
              hidden_activation=nn.Tanh(), output_activation=None):
    network = AttCNN(input_channels, output_dim, hidden_units,
          hidden_activation, output_activation)
    return network

def calculate_log_pi(log_stds, noises, actions):
    gaussian_log_probs = (-0.5 * noises.pow(2) - log_stds).sum(
        dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)

    return gaussian_log_probs - torch.log(
        1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)


def reparameterize(means, log_stds):
    noises = torch.randn_like(means)
    us = means + noises * log_stds.exp()
    actions = torch.tanh(us)
    return actions, calculate_log_pi(log_stds, noises, actions)


def atanh(x):
    return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))


def evaluate_lop_pi(means, log_stds, actions):
    noises = (atanh(actions) - means) / (log_stds.exp() + 1e-8)
    return calculate_log_pi(log_stds, noises, actions)
