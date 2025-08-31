import torch

class RolloutBuffer:
    def __init__(self, obs_dim, act_dim, horizon, num_envs, device):
        self.obs = torch.zeros(horizon+1, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(horizon, num_envs, act_dim, device=device)
        self.logprobs = torch.zeros(horizon, num_envs, device=device)
        self.rewards = torch.zeros(horizon, num_envs, device=device)
        self.dones = torch.zeros(horizon, num_envs, device=device, dtype=torch.bool)
        self.values = torch.zeros(horizon+1, num_envs, device=device)
        self.ptr = 0
        self.horizon = horizon
        self.num_envs = num_envs
        self.device = device

    def add(self, obs, action, logprob, reward, done, value):
        t = self.ptr
        self.obs[t].copy_(obs)
        self.actions[t].copy_(action)
        self.logprobs[t].copy_(logprob)
        self.rewards[t].copy_(reward)
        self.dones[t].copy_(done)
        self.values[t].copy_(value)
        self.ptr += 1

    def add_last_value(self, last_value, last_obs):
        self.obs[self.ptr].copy_(last_obs)
        self.values[self.ptr].copy_(last_value)

    def compute_gae(self, gamma, lam):
        T, N = self.horizon, self.num_envs
        adv = torch.zeros(T, N, device=self.device)
        last_gae = torch.zeros(N, device=self.device)
        for t in reversed(range(T)):
            nonterminal = (~self.dones[t]).float()
            delta = self.rewards[t] + gamma * self.values[t+1] * nonterminal - self.values[t]
            last_gae = delta + gamma * lam * nonterminal * last_gae
            adv[t] = last_gae
        returns = adv + self.values[:-1]
        # flatten
        obs = self.obs[:-1].reshape(T*N, -1)
        actions = self.actions.reshape(T*N, -1)
        logprobs = self.logprobs.reshape(T*N)
        returns = returns.reshape(T*N)
        adv = adv.reshape(T*N)
        values = self.values[:-1].reshape(T*N)
        # normalize advantages (stable)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return obs, actions, logprobs, returns, adv, values

