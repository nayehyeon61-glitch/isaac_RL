# ppo/entropy_ppo.py
import torch
import torch.nn as nn
from torch.optim import Adam


class EntropyPPO:
    """
    PPO + (선택) adaptive entropy(alpha)
    - self.actor_critic: evaluate_actions / act_fep 인터페이스 필요
    - rollouts: feed_forward_generator / returns/value_preds 등 제공
    """
    def __init__(self,
                 actor_critic,
                 clip_param=0.2,
                 ppo_epoch=4,
                 num_mini_batch=4,
                 value_loss_coef=0.5,
                 entropy_coef=0.0,
                 lr=3e-4,
                 max_grad_norm=0.5,
                 adaptive_entropy=False,
                 target_entropy=-1.0):
        self.actor_critic = actor_critic
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.optimizer = Adam(self.actor_critic.parameters(), lr=lr, eps=1e-5)

        # adaptive entropy
        self.adaptive_entropy = adaptive_entropy
        self.target_entropy = target_entropy
        if self.adaptive_entropy:
            self.alpha = nn.Parameter(
                torch.tensor(0.1, device=next(actor_critic.parameters()).device)
            )
            self.alpha_optimizer = Adam([self.alpha], lr=lr)
