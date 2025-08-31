import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

class PPO:
    def __init__(self, actor_critic, lr, clip_coef=0.2, vf_coef=0.5, ent_coef=0.01,
                 max_grad_norm=0.5, update_epochs=4, mini_batch_size=2048):
        self.ac = actor_critic
        self.opt = Adam(self.ac.parameters(), lr=lr)
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.mini_batch_size = mini_batch_size

    def update(self, obs, actions, old_logprobs, returns, adv, old_values):
        num_samples = obs.size(0)
        idx = torch.randperm(num_samples, device=obs.device)
        obs, actions, old_logprobs, returns, adv, old_values = (
            obs[idx], actions[idx], old_logprobs[idx], returns[idx], adv[idx], old_values[idx]
        )

        info = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        count = 0

        for _ in range(self.update_epochs):
            for start in range(0, num_samples, self.mini_batch_size):
                end = start + self.mini_batch_size
                b_obs = obs[start:end]
                b_act = actions[start:end]
                b_old_logp = old_logprobs[start:end]
                b_ret = returns[start:end]
                b_adv = adv[start:end]

                new_logp, ent, value = self.ac.evaluate_actions(b_obs, b_act)
                ratio = (new_logp - b_old_logp).exp()

                # Policy loss (clipped surrogate)
                unclipped = ratio * b_adv
                clipped = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * b_adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                # Value loss (clipped)
                value_clipped = old_values[start:end] + torch.clamp(
                    value - old_values[start:end], -self.clip_coef, self.clip_coef
                )
                v_unclipped = (value - b_ret).pow(2)
                v_clipped = (value_clipped - b_ret).pow(2)
                value_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()

                entropy_loss = -ent.mean()  # we add entropy *coef later as (+)

                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * (-entropy_loss)

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.opt.step()

                info["policy_loss"] += policy_loss.item()
                info["value_loss"] += value_loss.item()
                info["entropy"] += ent.mean().item()
                count += 1

        for k in info:
            info[k] /= max(count, 1)
        return info

