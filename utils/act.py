from .distributions import DiagGaussian, BetaDistr
import torch
import torch.nn as nn

class ACTLayer(nn.Module):
    def __init__(self, inputs_dim, action_dim, use_beta):
        super(ACTLayer, self).__init__()
        if use_beta:
            self.action_out = BetaDistr(inputs_dim, action_dim)
        else:
            self.action_out = DiagGaussian(inputs_dim, action_dim)
    
    def forward(self, x):
        action_logit = self.action_out(x)
        actions = action_logit.sample()
        # actions = torch.clamp(actions, -1.0, 1.0)
        action_log_probs = action_logit.log_probs(actions)
        return actions, action_log_probs

    def evaluate_actions(self, x, action):
        action_logit = self.action_out(x)
        action_log_probs = action_logit.log_probs(action)
        dist_entropy = action_logit.entropy().mean()
        return action_log_probs, dist_entropy
