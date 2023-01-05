import torch
from algorithm.actor_critic import actor, critic

class PPO:
    def __init__(self, args, critic_obs_space, actor_obs_space, action_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args['lr']
        self.critic_lr = args['critic_lr']
        self.opti_eps = args['opti_eps']
        self.use_beta = args['use_beta']
        self.critic_obs_space = critic_obs_space
        self.actor_obs_space = actor_obs_space
        self.action_space = action_space

        self.actor = actor(actor_obs_space.shape[0], action_space.shape[0], self.use_beta, device)
        self.critic = critic(critic_obs_space.shape[0], device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr,
                                                eps=self.opti_eps)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                lr=self.critic_lr,
                                                eps=self.opti_eps)

    def get_actions(self, critic_obs, actor_obs):
        actions, action_log_probs = self.actor(actor_obs)
        values = self.critic(critic_obs)
        return values, actions, action_log_probs

    def get_values(self, obs):
        values = self.critic(obs)
        return values

    def evaluate_action(self, critic_obs, actor_obs, action):
        action_log_probs, dist_entropy = self.actor.evaluate_action(actor_obs, action)
        values = self.critic(critic_obs)
        return values, action_log_probs, dist_entropy

    def act(self, obs):
        actions, _ = self.actor(obs)
        return actions

                
