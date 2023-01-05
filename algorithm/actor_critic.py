import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.act import ACTLayer
from utils.util import check

class actor(nn.Module):
    def __init__(self, input_dim, action_dim, use_beta=False, device=torch.device("cpu")):
        super(actor, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.fc = nn.Linear(input_dim, 64)
        self.act = ACTLayer(64, self.action_dim, use_beta)
        self.to(device)
        self.tpdv = dict(dtype=torch.float32, device=device)

    def forward(self, x):
        x = check(x).to(**self.tpdv)
        x = self.fc(x)
        x = F.relu(x)
        actions, action_log_probs = self.act(x)
        return actions, action_log_probs

    def evaluate_action(self, x, action):
        x = check(x).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        x = self.fc(x)
        x = F.relu(x)
        action_log_probs, dist_entropy = self.act.evaluate_actions(x, action)
        return action_log_probs, dist_entropy


class critic(nn.Module):
    def __init__(self, input_dim, device=torch.device("cpu")):
        super(critic, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.to(device)
        self.tpdv = dict(dtype=torch.float32, device=device)

    def forward(self, x):
        x = check(x).to(**self.tpdv)
        x = F.relu(self.fc1(x))
        values = self.fc2(x)
        return values
