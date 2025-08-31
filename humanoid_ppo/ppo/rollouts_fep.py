# ppo/rollouts_fep.py
import torch
import torch.nn as nn
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class PPORollouts:
    """
    Rollout storage for PPO (non-recurrent).
    Shapes (convention):
      - obs:              [T+1, N, obs_dim_raw]     ← 항상 '정규화된 원 관측'만 저장
      - actions:          [T,   N, act_dim]
      - action_log_probs: [T,   N, 1]
      - value_preds:      [T+1, N]
      - returns:          [T+1, N]
      - rewards:          [T,   N]
      - masks:            [T+1, N, 1]
    """
    def __init__(self, obs_dim, act_dim, horizon, num_envs, device):
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.horizon = int(horizon)
        self.num_envs = int(num_envs)
        self.device = torch.device(device)

        T, N, D, A = self.horizon, self.num_envs, self.obs_dim, self.act_dim

        self.obs = torch.zeros(T + 1, N, D, device=self.device)
        self.actions = torch.zeros(T, N, A, device=self.device)
        self.action_log_probs = torch.zeros(T, N, 1, device=self.device)
        self.value_preds = torch.zeros(T + 1, N, device=self.device)
        self.returns = torch.zeros(T + 1, N, device=self.device)
        self.rewards = torch.zeros(T, N, device=self.device)
        self.masks = torch.ones(T + 1, N, 1, device=self.device)

        self.step = 0

    @torch.no_grad()
    def insert(self, next_obs, rnn_states, actions, action_log_probs, value_preds, rewards, masks):
        """
        모든 입력은 torch.Tensor라고 가정.
        차원/모양이 조금 달라도 안전하게 맞춰 넣습니다.
        """
        t = self.step
        # obs: [N, D]
        if next_obs.dim() == 1:
            next_obs = next_obs.unsqueeze(0)
        self.obs[t + 1].copy_(next_obs)

        # actions: [N, A]
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)  # [N,1]
        self.actions[t].copy_(actions)

        # action_log_probs → [N, 1]
        lp = action_log_probs
        if lp.dim() == 1:
            lp = lp.unsqueeze(-1)         # [N,1]
        # 만약 [N, A]로 들어오면 합쳐서 한 스칼라로
        if lp.dim() == 2 and lp.shape[-1] != 1:
            lp = lp.sum(-1, keepdim=True) # [N,1]
        # 혹시 [N,1,1] 같은 꼴이면 squeeze
        if lp.dim() == 3 and lp.shape[-1] == 1:
            lp = lp.squeeze(-1)           # [N,1]
        self.action_log_probs[t].copy_(lp)

        # value_preds: [N] or [N,1] -> [N]
        v = value_preds
        if v.dim() == 2 and v.shape[-1] == 1:
            v = v.squeeze(-1)
        self.value_preds[t].copy_(v)

        # rewards: [N] or [N,1] -> [N]
        r = rewards
        if r.dim() == 2 and r.shape[-1] == 1:
            r = r.squeeze(-1)
        self.rewards[t].copy_(r)

        # masks: [N,1] (keep)
        if masks.dim() == 1:
            masks = masks.unsqueeze(-1)
        self.masks[t + 1].copy_(masks)

        self.step = (self.step + 1) % self.horizon

    @torch.no_grad()
    def after_update(self):
        # 마지막 관측을 0번째로 이동
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])
        self.step = 0

    @torch.no_grad()
    def compute_returns(self, next_value, gamma, gae_lambda):
        """
        GAE(λ) returns 계산.
        next_value: [N] 혹은 [N,1]
        """
        if next_value.dim() == 2 and next_value.shape[-1] == 1:
            next_value = next_value.squeeze(-1)

        self.value_preds[-1].copy_(next_value)
        gae = torch.zeros(self.num_envs, device=self.device)
        for t in reversed(range(self.horizon)):
            delta = self.rewards[t] + gamma * self.value_preds[t + 1] * self.masks[t + 1].squeeze(-1) - self.value_preds[t]
            gae = delta + gamma * gae_lambda * self.masks[t + 1].squeeze(-1) * gae
            self.returns[t] = gae + self.value_preds[t]

    def feed_forward_generator(self, advantages, num_mini_batch):
        """
        advantages: [T, N]
        """
        T, N = self.horizon, self.num_envs
        batch_size = T * N
        mini_batch_size = batch_size // num_mini_batch

        # 평탄화
        obs = self.obs[:-1].reshape(T * N, -1)
        actions = self.actions.reshape(T * N, -1)
        value_preds = self.value_preds[:-1].reshape(T * N)
        returns = self.returns[:-1].reshape(T * N)
        masks = self.masks[1:].reshape(T * N, -1)
        old_action_log_probs = self.action_log_probs.reshape(T * N, -1)
        adv = advantages.reshape(T * N)

        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True)
        for idx in sampler:
            yield (
                obs[idx],
                None,  # rnn states placeholder
                actions[idx],
                value_preds[idx],
                returns[idx],
                masks[idx],
                old_action_log_probs[idx].squeeze(-1),  # [B]
                adv[idx],
            )

    # rnn 버전이 필요하면 동일 패턴으로 추가 가능
