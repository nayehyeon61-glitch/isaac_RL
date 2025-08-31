# ppo/rollouts_fep.py
import torch


class PPORollouts:
    """
    IsaacGymEnvs VecTask 호환 PPO 롤아웃 버퍼
    - evaluate_actions/act_fep 시그니처에 맞춰 feed_forward_generator 제공
    """
    def __init__(self, obs_dim, act_dim, num_steps, num_envs, device):
        self.obs = torch.zeros(num_steps + 1, num_envs, obs_dim, device=device)
        self.recurrent_hidden_states = torch.zeros(num_steps + 1, num_envs, 1, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, 1, device=device)
        self.value_preds = torch.zeros(num_steps + 1, num_envs, 1, device=device)
        self.returns = torch.zeros(num_steps + 1, num_envs, 1, device=device)
        self.action_log_probs = torch.zeros(num_steps, num_envs, 1, device=device)
        self.actions = torch.zeros(num_steps, num_envs, act_dim, device=device)
        self.masks = torch.ones(num_steps + 1, num_envs, 1, device=device)
        self.step = 0
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

    def insert(self, obs, rnn_states, actions, action_log_probs, value_preds, rewards, masks):
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step + 1].copy_(rnn_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs.unsqueeze(-1))
        self.value_preds[self.step].copy_(value_preds.unsqueeze(-1))
        self.rewards[self.step].copy_(rewards.unsqueeze(-1))
        self.masks[self.step + 1].copy_(masks)
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.step = 0

    def compute_returns(self, next_value, gamma, gae_lambda):
        self.value_preds[-1] = next_value.unsqueeze(-1)
        gae = 0
        for step in reversed(range(self.rewards.size(0))):
            delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
            gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + self.value_preds[step]

    def feed_forward_generator(self, advantages, num_mini_batch):
        num_steps, num_envs = self.rewards.size(0), self.rewards.size(1)
        batch_size = num_envs * num_steps
        mini_batch_size = batch_size // num_mini_batch

        obs = self.obs[:-1].reshape(batch_size, -1)
        actions = self.actions.reshape(batch_size, -1)
        value_preds = self.value_preds[:-1].reshape(batch_size, 1)
        returns = self.returns[:-1].reshape(batch_size, 1)
        masks = self.masks[:-1].reshape(batch_size, 1)
        old_action_log_probs = self.action_log_probs.reshape(batch_size, 1).squeeze(-1)
        adv_targ = advantages.reshape(batch_size, 1).squeeze(-1)

        perm = torch.randperm(batch_size, device=self.device)
        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            idx = perm[start:end]
            yield (obs[idx],
                   torch.zeros_like(idx, dtype=obs.dtype, device=self.device).unsqueeze(-1),  # dummy rnn
                   actions[idx],
                   value_preds[idx].squeeze(-1),
                   returns[idx].squeeze(-1),
                   masks[idx],
                   old_action_log_probs[idx],
                   adv_targ[idx])
