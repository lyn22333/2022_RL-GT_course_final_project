import numpy as np
import torch
import vmas
from tensorboardX import SummaryWriter
from utils.buffer import SharedReplayBuffer

def _t2n(x):
    return x.detach().cpu().numpy()


class MAPPO_Runner(object):

    def __init__(self, args, env):
        self.args = args
        self.envs = env
        self.num_envs = args["num_envs"]
        self.n_agents = args['n_agents']
        self.obs_space = args['obs_space']
        self.cent_obs_space = args['cent_obs_space']
        self.action_space = args['action_space']
        self.iterations = args['iterations']
        self.episode_length = args['episode_length']
        self.device = args['device']
        self.eval_interval = args['eval_interval']
        self.use_beta = args['use_beta']
        self.log_dir = args['log_dir'] + '/MAPPO' + '_' + str(self.n_agents)
        # self.log_dir = args['log_dir'] + '/MAPPO'
        if self.use_beta:
            self.log_dir += '_Beta'
        self.log_dir = 'log/test'
        self.log_file = open(self.log_dir + '/reward.txt', 'w+')
        self.writer = SummaryWriter(self.log_dir)

        from algorithm.trainer import Trainer as TrainAlgo
        from algorithm.ppo import PPO as Policy

        # policy network
        self.policy = Policy(self.args,
                            actor_obs_space=self.obs_space,
                            critic_obs_space = self.cent_obs_space,
                            action_space = self.action_space,
                            device = self.device)

        # algorithm
        self.trainer = TrainAlgo(self.args, self.policy, device = self.device)
        
        # buffer
        self.buffer = SharedReplayBuffer(self.args,
                                        self.n_agents,
                                        actor_obs_space=self.obs_space,
                                        critic_obs_space = self.cent_obs_space,
                                        act_space = self.action_space)

    def run(self):
        for iteration in range(self.iterations):
            self.warmup()
            train_total_reward = 0
            last_done = torch.tensor([False for _ in range(self.num_envs)])
            for step in range(self.episode_length):
                # (num_envs, n_agents, -1)
                values, actions, action_log_probs = self.collect(step)

                # (n_agents, num_envs, -1)
                actions_env = np.swapaxes(actions, 0, 1)
                actions_env = torch.tensor(actions_env)
                if self.use_beta:
                    actions_env = 2.0*(actions_env-0.5)
                actions_env = torch.clamp(actions_env, -1.0, 1.0)
                obs, rewards, dones, infos = self.envs.step(actions_env)

                train_rewards = rewards[0]
                mean_global_reward = train_rewards[last_done==False]
                train_total_reward += mean_global_reward.mean()
                last_done = dones

                # (num_envs, n_agents, -1)
                obs = torch.stack(obs).numpy()
                obs = np.swapaxes(obs, 0, 1)
                rewards = torch.stack(rewards).numpy()
                rewards = np.swapaxes(rewards, 0, 1)
                rewards = np.reshape(rewards, (self.num_envs, self.n_agents, 1))

                dones = np.expand_dims(dones.numpy(),1).repeat(self.n_agents, 1)

                data = obs, rewards, dones, infos, values, actions, action_log_probs
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            train_infos['train_episode_reward'] = train_total_reward
            self.log_file.write(f"iteration: {iteration}, rewards:{round(train_total_reward.item(), 3)}\n")
            self.log_file.flush()
            self.log(train_infos, iteration)

    def warmup(self):
        obs = self.envs.reset()
        obs = torch.stack(obs).numpy()
        obs = np.swapaxes(obs, 0, 1)
        share_obs = obs.reshape(self.num_envs, -1)
        share_obs = np.expand_dims(share_obs, 1).repeat(self.n_agents, axis=1)
        self.buffer.critic_obs[0] = share_obs.copy()
        self.buffer.actor_obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob = self.trainer.policy.get_actions(np.concatenate(self.buffer.critic_obs[step]), 
                                                                            np.concatenate(self.buffer.actor_obs[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.num_envs))
        actions = np.array(np.split(_t2n(action), self.num_envs))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.num_envs))
        return values, actions, action_log_probs
    
    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs = data
        masks = np.ones((self.num_envs, self.n_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        share_obs = obs.reshape(self.num_envs, -1)
        share_obs = np.expand_dims(share_obs, 1).repeat(self.n_agents, axis=1)
        critic_obs = share_obs
        actor_obs = obs
        self.buffer.insert(critic_obs, actor_obs, actions, action_log_probs, values, rewards, masks)

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.critic_obs[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.num_envs))
        self.buffer.compute_returns(next_values)

    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)      
        self.buffer.after_update()
        return train_infos
 
    def log(self, train_infos, iteration):
        print(iteration)
        for k, v in train_infos.items():
            print(k, v)
            self.writer.add_scalar(k, v, iteration)


env = vmas.make_env("wheel", 300, "cpu", True)
obs_space = env.observation_space[0]
from gym.spaces.box import Box
shape = (obs_space.shape[0]*env.n_agents,)
low = np.array([-np.inf for _ in range(shape[0])])
high = np.array([np.inf for _ in range(shape[0])])
cent_obs_space = Box(low, high, shape, np.float64)
action_space = env.action_space[0]
arg = {
    "num_envs": 300,
    "n_agents": env.n_agents,
    "obs_space": obs_space,
    "action_space": action_space,
    "iterations": 400,
    "episode_length": 200,
    "cent_obs_space": cent_obs_space,
    "device": "cpu",
    "eval_interval": 5,
    "log_dir": "./log",

    "gamma": 0.99,
    "gae_lambda": 0.9,
    "lr": 5e-5,
    "critic_lr": 5e-5,
    "opti_eps": 1e-5,
    "clip_param": 0.2,
    "ppo_epoch": 15,
    "batch_size": 4000,
    "max_grad_norm": 0.5,

    "use_beta": False,
}

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

runner = MAPPO_Runner(arg, env)
runner.run()