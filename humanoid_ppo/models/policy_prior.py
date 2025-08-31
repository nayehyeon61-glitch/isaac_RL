# models/policy_prior.py
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform


class PolicyPrior(nn.Module):
    """
    p_prior(a|s): tanh-Gaussian
    - obs -> (mu, log_std)로 매핑, TransformedDistribution으로 log_prob 계산
    """
    def __init__(self, obs_dim, act_dim, hidden_size=256, init_log_std=0.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
        )
        self.mu = nn.Linear(hidden_size, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))

    def dist(self, obs):
        obs = torch.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)
        h = self.body(obs)
        mu = self.mu(h)
        mu = torch.nan_to_num(mu, nan=0.0, posinf=100.0, neginf=-100.0)
        std = self.log_std.exp().clamp(min=1e-6, max=10.0)
        base = Normal(mu, std)
        return TransformedDistribution(base, [TanhTransform(cache_size=1)])

    @torch.no_grad()
    def sample(self, obs):
        d = self.dist(obs)
        return d.sample()

    def log_prob(self, obs, actions):
        d = self.dist(obs)
        return d.log_prob(actions).sum(-1)
