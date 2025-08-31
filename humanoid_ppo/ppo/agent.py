# ppo/agent.py
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=256):
        super().__init__()
        # Actor
        self.pi = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden_size, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # Critic
        self.vf = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    is_recurrent = False  # FEP PPO 인터페이스 호환 용

    def _dist(self, obs):
        # 입력 세척(배치에서 NaN/Inf 방지)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)
        h = self.pi(obs)
        mu = self.mu_head(h)
        mu = torch.nan_to_num(mu, nan=0.0, posinf=100.0, neginf=-100.0)
        std = self.log_std.exp().clamp(min=1e-6, max=10.0)
        base = Normal(mu, std)
        return TransformedDistribution(base, [TanhTransform(cache_size=1)])

    @torch.no_grad()
    def act_fep(self, obs, rnn_states, masks):
        dist = self._dist(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        value = self.vf(obs).squeeze(-1)
        return value, action, log_prob, rnn_states

    def evaluate_actions(self, obs, rnn_states, masks, actions):
        dist = self._dist(obs)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.base_dist.entropy().sum(-1)  # tanh 이전 base entropy
        value = self.vf(obs).squeeze(-1)
        return value, log_prob, entropy, None
