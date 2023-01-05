import numpy as np
import torch
import torch.nn as nn
from utils.util import check

class Trainer():
    def __init__(self, args, policy, device=torch.device("cpu")):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args['clip_param']
        self.ppo_epoch = args['ppo_epoch']
        self.batch_size = args['batch_size']
        self.max_grad_norm = args['max_grad_norm']

    def mse_loss(self, e):
        return e**2/2

    def cal_value_loss(self, values, value_preds_batch, return_batch):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                    self.clip_param)
        error_clipped = return_batch - value_pred_clipped
        error_original = return_batch - values
        value_loss_clipped = self.mse_loss(error_clipped)
        value_loss_original = self.mse_loss(error_original)
        value_loss = torch.max(value_loss_original, value_loss_clipped)
        return value_loss.mean()

    def ppo_update(self, sample):
        critic_obs_batch, actor_obs_batch, actions_batch, \
        value_preds_batch, return_batch, old_action_log_probs_batch, \
        adv_targ = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)

        values, action_log_probs, dist_entropy = self.policy.evaluate_action(critic_obs_batch, actor_obs_batch, actions_batch)

        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        policy_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()
        self.policy.actor_optimizer.zero_grad()
        (policy_loss - 0.01*dist_entropy).backward()
        nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        self.policy.actor_optimizer.step()

        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch)
        self.policy.critic_optimizer.zero_grad()
        (0.5*value_loss).backward()
        nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        self.policy.critic_optimizer.step()

        return value_loss, policy_loss, dist_entropy, imp_weights

    def train(self, buffer):
        advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['ratio'] = 0
        num_updates = 0

        for _ in range(self.ppo_epoch):
            data_generator = buffer.feed_forward_generator(advantages, self.batch_size)
            for sample in data_generator:
                num_updates += 1
                value_loss, policy_loss, dist_entropy, imp_weights = self.ppo_update(sample)
                train_info['value_loss'] += value_loss
                train_info['policy_loss'] += policy_loss
                train_info['dist_entropy'] += dist_entropy
                train_info['ratio'] += imp_weights.mean()

        for k in train_info.keys():
            train_info[k]/=num_updates

        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
