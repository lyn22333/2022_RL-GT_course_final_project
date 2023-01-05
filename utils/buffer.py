import torch
import numpy as np

class SharedReplayBuffer(object):
    def __init__(self, args, num_agents, critic_obs_space, actor_obs_space, act_space):
        self.episode_length = args['episode_length']
        self.num_envs = args['num_envs']
        self.gamma = args['gamma']
        self.gae_lambda = args['gae_lambda']
        critic_obs_space = critic_obs_space.shape
        actor_obs_space = actor_obs_space.shape

        self.critic_obs = np.zeros((self.episode_length + 1, self.num_envs, num_agents, *critic_obs_space), dtype=np.float32)
        self.actor_obs = np.zeros((self.episode_length + 1, self.num_envs, num_agents, *actor_obs_space), dtype=np.float32)

        self.value_preds = np.zeros(
            (self.episode_length + 1, self.num_envs, num_agents, 1), dtype=np.float32)
        self.returns = np.zeros_like(self.value_preds)
        self.masks = np.ones((self.episode_length + 1, self.num_envs, num_agents, 1), dtype=np.float32)

        act_shape = act_space.shape

        self.actions = np.zeros(
            (self.episode_length, self.num_envs, num_agents, act_shape[0]), dtype=np.float32)
        self.action_log_probs = np.zeros(
            (self.episode_length, self.num_envs, num_agents, act_shape[0]), dtype=np.float32)
        self.rewards = np.zeros(
            (self.episode_length, self.num_envs, num_agents, 1), dtype=np.float32)

        self.step = 0

    def insert(self, critic_obs, actor_obs, actions, action_log_probs, value_preds, rewards, masks):
        self.critic_obs[self.step + 1] = critic_obs.copy()
        self.actor_obs[self.step + 1] = actor_obs.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        self.critic_obs[0] = self.critic_obs[-1].copy()
        self.actor_obs[0] = self.actor_obs[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.step = 0

    def compute_returns(self, next_value):
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
                delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                        self.value_preds[step]
                gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]

    def feed_forward_generator(self, advantages, mini_batch_size):
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents

        num_mini_batch = batch_size // mini_batch_size
        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        critic_obs = self.critic_obs[:-1].reshape(-1, *self.critic_obs.shape[3:])
        actor_obs = self.actor_obs[:-1].reshape(-1, *self.actor_obs.shape[3:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            # obs size [T+1 N M Dim]-->[T N M Dim]-->[T*N*M,Dim]-->[index,Dim]
            critic_obs_batch = critic_obs[indices]
            actor_obs_batch = actor_obs[indices]
            actions_batch = actions[indices]
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            yield critic_obs_batch, actor_obs_batch, actions_batch, value_preds_batch, \
                    return_batch, old_action_log_probs_batch, adv_targ