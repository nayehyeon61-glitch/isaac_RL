# ppo/fep_ppo.py
import torch
import torch.nn as nn
import torch.optim as optim
from .entropy_ppo import EntropyPPO


class FEP_PPO(EntropyPPO):
    """
    EntropyPPO + Free-Energy 기반 policy-prior 정규화
    - λ * KL( posterior || prior ) = E[ logπ(a|s) - log p_prior(a|s) ]
    - (옵션) prior를 전문가 데이터로 동시학습(NLL 최소화)
    """
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
        self.demos_iter = iter(demos_loader) if demos_loader is not None else None
        self.demos_steps_per_update = int(demos_steps_per_update)
        self.prior_optim = optim.Adam(self.policy_prior.parameters(), lr=prior_lr, eps=1e-5)

    def _kl_post_prior(self, obs, actions, action_log_probs):
        """E[ logπ(a|s) - log p_prior(a|s) ]  (배치 평균 스칼라로 반환)"""
        with torch.no_grad():
            logp_post = action_log_probs
        logp_prior = self.policy_prior.log_prob(obs, actions)
        diff = logp_post - logp_prior
        return diff.mean()

    def _update_policy_prior_with_demos(self, device):
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
            obs = torch.nan_to_num(obs.to(device), nan=0.0, posinf=100.0, neginf=-100.0)
            act = torch.nan_to_num(act.to(device), nan=0.0, posinf=1.0, neginf=-1.0)
            nll = - self.policy_prior.log_prob(obs, act).mean()
            self.prior_optim.zero_grad()
            nll.backward()
            nn.utils.clip_grad_norm_(self.policy_prior.parameters(), 1.0)
            self.prior_optim.step()
            logs["prior_nll"] = float(nll.item())
        return logs

    def update(self, rollouts):
        # GAE는 rollouts.compute_returns()에서 미리 계산했다고 가정
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0.0
        policy_loss_epoch = 0.0
        entropy_epoch = 0.0
        kl_epoch = 0.0
        alpha_value_epoch = 0.0

        device = next(self.actor_critic.parameters()).device

        for _ in range(self.ppo_epoch):
            if getattr(self.actor_critic, "is_recurrent", False):
                data_generator = rollouts.recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                (obs_batch, rnn_states_batch, actions_batch,
                 value_preds_batch, return_batch, masks_batch,
                 old_action_log_probs_batch, adv_targ) = sample

                # Posterior 평가
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, rnn_states_batch, masks_batch, actions_batch
                )

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                policy_loss = -torch.min(surr1, surr2).mean()

                # clipped value loss
                value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (values - return_batch).pow(2)
                value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                # FEP 정규화 (스칼라)
                kl_val = self._kl_post_prior(obs_batch, actions_batch, action_log_probs)

                # 엔트로피 평균 스칼라
                ent_mean = dist_entropy.mean()

                # 총손실 (스칼라 보장)
                if self.adaptive_entropy:
                    entropy_term = (self.alpha.detach() * action_log_probs).mean()
                    total_loss = (
                        value_loss * self.value_loss_coef
                        + policy_loss
                        + entropy_term
                        + self.prior_coef * kl_val
                    )
                else:
                    total_loss = (
                        value_loss * self.value_loss_coef
                        + policy_loss
                        - self.entropy_coef * ent_mean
                        + self.prior_coef * kl_val
                    )

                if total_loss.dim() > 0:
                    total_loss = total_loss.mean()

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                if self.adaptive_entropy:
                    with torch.no_grad():
                        neg_logp = (-action_log_probs).detach()
                    alpha_loss = (self.alpha * (neg_logp - self.target_entropy)).mean()
                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                    alpha_value_epoch += float(self.alpha.item())

                # 로깅(스칼라만 누적)
                value_loss_epoch += float(value_loss.item())
                policy_loss_epoch += float(policy_loss.item())
                entropy_epoch += float(ent_mean.item())
                kl_epoch += float(kl_val.item())

        prior_logs = self._update_policy_prior_with_demos(device)

        num_updates = self.ppo_epoch * self.num_mini_batch
        logs = {
            "value_loss": value_loss_epoch / num_updates,
            "policy_loss": policy_loss_epoch / num_updates,
            "entropy": entropy_epoch / num_updates,
            "kl_post_prior": kl_epoch / num_updates,
        }
        if self.adaptive_entropy:
            logs["alpha"] = alpha_value_epoch / num_updates
        logs.update(prior_logs)
        return logs
