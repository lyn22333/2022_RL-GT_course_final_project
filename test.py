import numpy as np
import torch


dist = torch.distributions.Beta(100.0, 1.0001)
print(dist.log_prob(torch.tensor(0.9999999)))