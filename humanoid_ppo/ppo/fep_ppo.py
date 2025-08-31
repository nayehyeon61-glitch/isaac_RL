# ppo/fep_ppo.py
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any

class EntropyPPO(nn.Module):
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
        super().__init__()
        self.actor_critic = actor_critic
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr, eps=1e-5)

        self.adaptive_entropy = adaptive_entropy
        if adaptive_entropy:
            self.alpha = nn.Parameter(torch.tensor(0.0))
            self.target_entropy = float(target_entropy)
            self.alpha_optimizer = optim.Adam([self.alpha], lr=lr, eps=1e-5)

    def update(self, *args, **kwargs):
        raise NotImplementedError


class FEP_PPO(EntropyPPO):
    def __init__(self,
                 actor_critic,
                 policy_prior,
                 prior_coef: float = 1.0,
                 prior_lr: float = 3e-4,
                 demos_loader=None,
                 demos_steps_per_update: int = 0,
                 *args, **kwargs):
        super().__init__(actor_critic, *args, **kwargs)
        self.policy_prior = policy_prior
        self.prior_coef = float(prior_coef)
        self.demos_loader = demos_loader
        self.demos_steps_per_update = int(demos_steps_per_update)
        self.demos_iter = iter(demos_loader) if demos_loader is not None else None
        self.prior_optim = optim.Adam(self.policy_prior.parameters(), lr=prior_lr, eps=1e-5)

    @torch.no_grad()
    def _posterior_logp_detached(self, action_log_probs: torch.Tensor) -> torch.Tensor:
        return action_log_probs.detach()

    def _kl_post_prior(self, obs, actions, action_log_probs) -> torch.Tensor:
        """
        KL(post || prior) = E[ logπ(a|s) - log p_prior(a|s) ]
        - prior 입력도 actor와 동일한 피처로 맞춤: feats = actor_critic._prep(obs).detach()
        - posterior 항은 detach (actor 그래프 차단)
        """
        with torch.no_grad():
            feats = self.actor_critic._prep(obs)  # [B, obs+2*z]
        feats = feats.detach()
        logp_post = self._posterior_logp_detached(action_log_probs)  # [B]
        logp_prior = self.policy_prior.log_prob(feats, actions.detach())  # [B]
        return (logp_post - logp_prior).mean()

    def _update_policy_prior_with_demos(self, device) -> Dict[str, Any]:
        logs = {}
        if self.demos_loader is None or self.demos_steps_per_update <= 0:
            return logs
        self.policy_prior.train()
        for _ in range(self.demos_steps_per_update):
            try:
                obs, act = next(self.demos_iter)
            except StopIteration:
                self.demos_iter = iter(self.demos_loader)
                obs, act = next(self.demos_iter)
            obs = obs.to(device)
            act = act.to(device)
            nll = - self.policy_prior.log_prob(obs, act).mean()
            self.prior_optim.zero_grad(set_to_none=True)
            nll.backward()
            nn.utils.clip_grad_norm_(self.policy_prior.parameters(), 1.0)
            self.prior_optim.step()
            logs["prior_nll"] = float(nll.item())
        return logs

    def update(self, rollouts):
        adv = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-5)

        value_loss_epoch = action_loss_epoch = dist_entropy_epoch = kl_epoch = 0.0
        alpha_value_epoch = 0.0

        for _ in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(adv, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(adv, self.num_mini_batch)

            for sample in data_generator:
                (obs_b, rnn_b, act_b,
                 vpred_b, ret_b, masks_b,
                 old_logp_b, adv_b) = sample

                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_b, rnn_b, masks_b, act_b
                )
                if dist_entropy.dim() > 0:
                    dist_entropy = dist_entropy.mean()

                ratio = torch.exp(action_log_probs - old_logp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_b
                action_loss = -torch.min(surr1, surr2).mean()

                vpred_clip = vpred_b + (values - vpred_b).clamp(-self.clip_param, self.clip_param)
                vloss = 0.5 * torch.max((values - ret_b).pow(2), (vpred_clip - ret_b).pow(2)).mean()

                kl = self._kl_post_prior(obs_b, act_b, action_log_probs)

                if self.adaptive_entropy:
                    entropy_term = (self.alpha.detach() * action_log_probs).mean()
                    total_loss = vloss * self.value_loss_coef + action_loss + entropy_term + self.prior_coef * kl
                else:
                    total_loss = vloss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef + self.prior_coef * kl

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                if self.adaptive_entropy:
                    with torch.no_grad():
                        neg_logp = (-action_log_probs).detach()
                    alpha_loss = (self.alpha * (neg_logp - self.target_entropy)).mean()
                    self.alpha_optimizer.zero_grad(set_to_none=True)
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                    alpha_value_epoch += float(self.alpha.item())

                value_loss_epoch += float(vloss.item())
                action_loss_epoch += float(action_loss.item())
                dist_entropy_epoch += float(dist_entropy.item()) if torch.is_tensor(dist_entropy) else float(dist_entropy)
                kl_epoch += float(kl.item())

                del values, action_log_probs, ratio, surr1, surr2, vpred_clip, vloss, kl, total_loss

        prior_logs = self._update_policy_prior_with_demos(next(self.actor_critic.parameters()).device)

        num_updates = max(1, self.ppo_epoch * self.num_mini_batch)
        logs: Dict[str, Any] = {
            "value_loss": value_loss_epoch / num_updates,
            "action_loss": action_loss_epoch / num_updates,
            "entropy": dist_entropy_epoch / num_updates,
            "kl_post_prior": kl_epoch / num_updates,
        }
        if self.adaptive_entropy:
            logs["alpha"] = alpha_value_epoch / num_updates
        if isinstance(prior_logs, dict):
            logs.update(prior_logs)
        return logs
