import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import init

class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(DiagGaussian, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())

class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        bias = self._bias.t().view(1, -1)
        return x + bias


class BetaDistr(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(BetaDistr, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.alpha_layer = init_(nn.Linear(num_inputs, num_outputs))
        self.beta_layer = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        alpha = F.softplus(self.alpha_layer(x)) + 1.0
        beta = F.softplus(self.beta_layer(x)) + 1.0
        alpha = torch.clamp(alpha, 1.0001, 100.0)
        beta = torch.clamp(beta, 1.0001, 100.0)
        return FixedBeta(alpha, beta)

class FixedBeta(torch.distributions.Beta):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
