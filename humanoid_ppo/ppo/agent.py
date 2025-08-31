# ppo/agent.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

def _safe(t, clip=100.0):
    t = torch.nan_to_num(t, nan=0.0, posinf=clip, neginf=-clip)
    return torch.clamp(t, -clip, clip)

class MLPActorCritic(nn.Module):
    """
    - feature_extractor가 있으면 obs -> [obs, z, z_next] 로 확장(내부에서만 수행)
    - pi/vf 모두 동일한 확장 차원을 사용
    """
    def __init__(self, obs_dim, act_dim, hidden_size=256, feature_extractor=None):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.is_recurrent = False  # 본 예제에선 RNN 미사용

        # 확장 후 실제 정책 입력 차원
        if self.feature_extractor is None:
            self.input_dim = int(obs_dim)
        else:
            # WMFeatureAdapter.z_dim을 신뢰 (obs_dim_raw + 2*z_dim)
            zdim = int(getattr(self.feature_extractor, "z_dim", 0))
            self.input_dim = int(obs_dim + 2 * zdim)

        self.act_dim = int(act_dim)

        # 정책 네트워크
        self.pi = nn.Sequential(
            nn.Linear(self.input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.mu_head = nn.Linear(hidden_size, self.act_dim)
        self.logstd = nn.Parameter(torch.ones(self.act_dim) * -0.5)

        # 가치함수
        self.vf_net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def _prep(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: [B, D_obs_raw]  →  x: [B, D_obs_raw + 2*z_dim] (feature_extractor 있을 때)
        """
        obs = _safe(obs).float()
        if self.feature_extractor is not None:
            x = self.feature_extractor(obs)  # [B, obs+2*z]
        else:
            x = obs

        # 안전 체크: 네트가 기대하는 입력 차원과 맞는지
        if x.shape[-1] != self.input_dim:
            raise RuntimeError(
                f"[MLPActorCritic._prep] feature 차원 불일치: got {x.shape[-1]}, expected {self.input_dim}. "
                f"(feature_extractor={'on' if self.feature_extractor is not None else 'off'})"
            )
        return x

    def _dist(self, x: torch.Tensor):
        h = self.pi(x)
        mu = self.mu_head(h)
        std = self.logstd.exp().expand_as(mu)
        return Normal(mu, std)

    @torch.no_grad()
    def act(self, obs, rnn_states, masks):
        """표준 PPO 행동(월드모델 없이 쓸 때). 여기선 사용 안 하지만 남겨둠."""
        x = self._prep(obs)
        dist = self._dist(x)
        action = torch.tanh(dist.rsample())
        log_prob = dist.log_prob(torch.atanh(torch.clamp(action, -0.999, 0.999))).sum(-1, keepdim=True)
        value = self.vf_net(x).squeeze(-1)
        return value, action, log_prob, rnn_states

    @torch.no_grad()
    def act_fep(self, obs, rnn_states, masks):
        """
        FEP-PPO에서 사용: 내부에서 항상 WMFeatureAdapter를 적용(_prep)해 확장 피처 사용
        """
        x = self._prep(obs)                      # ✅ [obs, z, z_next]
        dist = self._dist(x)
        # reparameterize + tanh squash
        pre_a = dist.rsample()
        action = torch.tanh(pre_a)
        # log_prob (tanh 정규화 역변환)
        atanh_a = torch.atanh(torch.clamp(action, -0.999, 0.999))
        log_prob = dist.log_prob(atanh_a) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        value = self.vf_net(x).squeeze(-1)
        return value, action, log_prob, rnn_states

    def evaluate_actions(self, obs, rnn_states, masks, actions):
        """
        rollouts에서 샘플된 (obs, actions)에 대해 log_prob, entropy, value 평가
        obs는 항상 'raw 정규화 관측'이어야 하고, 여기서 _prep으로 확장합니다.
        """
        x = self._prep(obs)                      # ✅ [obs, z, z_next]
        dist = self._dist(x)

        # tanh squashed action의 log_prob 재계산
        atanh_a = torch.atanh(torch.clamp(actions, -0.999, 0.999))
        log_prob = dist.log_prob(atanh_a) - torch.log(1 - actions.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)

        # 엔트로피(가우시안 기준; tanh 보정은 근사적으로 무시하거나 필요 시 추가)
        entropy = dist.entropy().sum(-1)  # [B]

        values = self.vf_net(x).squeeze(-1)
        return values, log_prob, entropy, rnn_states

    # 가치함수 직접 호출 시(bootstrap 등) 반드시 _prep를 거친 특징을 넣어야 함
    def vf(self, x_feat: torch.Tensor) -> torch.Tensor:
        return self.vf_net(_safe(x_feat).float())
