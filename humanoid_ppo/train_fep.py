# train_fep.py
# FEP-EntropyPPO + World Model feature integration
# 사용 예: python train_fep.py --config configs/humanoid_fep.yaml

import os, sys
import argparse
import yaml
import random
import numpy as np

# ⚠️ import 순서: isaacgym -> isaacgymenvs -> torch
from isaacgym import gymapi
import isaacgymenvs
import torch
from tqdm import trange

from tasks.make_env import make_isaac_env, extract_obs, sanitize_obs
from tasks.make_env import set_viewer_camera_close, follow_target_each_step
from ppo.agent import MLPActorCritic
from ppo.rollouts_fep import PPORollouts
from ppo.fep_ppo import FEP_PPO
from models.policy_prior import PolicyPrior
from models.wm_integration import WMFeatureAdapter


def set_seed(seed: int = 42):
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


class RunningNorm:
    """온라인 정규화 (x - mean) / sqrt(var + eps)"""
    def __init__(self, shape, eps=1e-5, device="cpu"):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = torch.tensor(eps, device=device)

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = torch.tensor(x.shape[0], device=x.device, dtype=x.dtype)
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / tot)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * (self.count * batch_count / tot)
        self.mean = new_mean
        self.var = M2 / tot
        self.count = tot

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg.get("seed", 1)))

    sim_device  = cfg.get("sim_device", None)
    rl_device   = cfg.get("rl_device", None)
    headless    = bool(cfg.get("headless", True))
    graphics_id = int(cfg.get("graphics_device_id", 0))
    render_enabled = bool(cfg.get("render", False))
    render_interval = int(cfg.get("render_interval", 1))

    # ── Isaac env
    env, device, obs_dim_raw, act_dim = make_isaac_env(
        task_name=cfg["task_name"],
        num_envs=int(cfg["num_envs"]),
        seed=int(cfg["seed"]),
        sim_device=sim_device,
        rl_device=rl_device,
        headless=headless,
        graphics_device_id=graphics_id,
    )
    if not headless:
        set_viewer_camera_close(env,env_index=0,distance=float(cfg.get("cam_distance",3.0)),
                                 height=float(cfg.get("cam_height",1.2)),
                                 yaw_deg=float(cfg.get("cam_yaw",30.0)))
    # ── WorldModel feature 연결(옵션) — 피처추출은 actor 내부에서만 수행
    wm_cfg = cfg.get("wm", {})
    use_wm = bool(wm_cfg.get("enabled", False))
    feature_extractor = None
    final_obs_dim = obs_dim_raw

    if use_wm:
        dof_dim = int(wm_cfg.get("dof_dim", 35))
        z_dim   = int(wm_cfg.get("z_dim", 64))
        ckpt    = str(wm_cfg.get("ckpt_path", "world_model_diffode.pt"))
        detach  = bool(wm_cfg.get("detach_z", False))

        feature_extractor = WMFeatureAdapter(
            obs_dim=obs_dim_raw,
            dof_dim=dof_dim,
            z_dim=z_dim,
            wm_ckpt_path=ckpt,
            device=device,
            allow_grad_through_adapter=True,
            detach_z=detach,
        ).to(device)
        final_obs_dim = obs_dim_raw + 2 * z_dim

    # ── Agent / Prior
    hidden_size = int(cfg.get("hidden_size", 256))
    ac = MLPActorCritic(obs_dim_raw, act_dim, hidden_size=hidden_size,
                    feature_extractor=feature_extractor).to(device)
    prior = PolicyPrior(final_obs_dim, act_dim, hidden_size=hidden_size).to(device)

    algo = FEP_PPO(
        actor_critic=ac,
        policy_prior=prior,
        prior_coef=float(cfg.get("prior_coef", 1.0)),
        prior_lr=float(cfg.get("prior_lr", 3e-4)),
        demos_loader=None,
        demos_steps_per_update=int(cfg.get("demos_steps_per_update", 0)),
        clip_param=float(cfg.get("clip_param", 0.2)),
        ppo_epoch=int(cfg.get("ppo_epoch", 4)),
        num_mini_batch=int(cfg.get("num_mini_batch", 4)),
        value_loss_coef=float(cfg.get("value_loss_coef", 0.5)),
        entropy_coef=float(cfg.get("entropy_coef", 0.01)),
        lr=float(cfg.get("learning_rate", 3e-4)),
        max_grad_norm=float(cfg.get("max_grad_norm", 0.5)),
        adaptive_entropy=bool(cfg.get("adaptive_entropy", False)),
        target_entropy=float(cfg.get("target_entropy", -1.0)),
    )

    horizon    = int(cfg.get("horizon", 1024))
    num_envs   = int(cfg.get("num_envs", 2))
    gamma      = float(cfg.get("gamma", 0.99))
    gae_lambda = float(cfg.get("gae_lambda", 0.95))

    # ── Rollouts: 항상 '정규화된 원 관측' 차원으로 생성
    rollouts = PPORollouts(obs_dim_raw, act_dim, horizon, num_envs, device)

    # ── 초기 관측 (월드모델 적용 금지: raw만 저장/정규화)
    obs_raw = env.reset()
    obs = sanitize_obs(extract_obs(obs_raw).to(device)).float()
    obs_rms = RunningNorm(obs.shape[-1], device=device)
    obs_rms.update(obs)
    obs_n = obs_rms.normalize(obs)
    rollouts.obs[0].copy_(obs_n)

    rnn_states = torch.zeros(num_envs, 1, device=device)
    masks = torch.ones(num_envs, 1, device=device)

    total_updates = int(cfg.get("total_updates", 1000))
    log_interval  = int(cfg.get("log_interval", 10))
    eval_interval = int(cfg.get("eval_interval", 50))

    for update in trange(total_updates, desc="FEP-PPO"):
        for step in range(horizon):
            with torch.no_grad():
                # ✅ actor 내부에서만 WMFeatureAdapter 적용됨
                values, actions, log_probs, rnn_states = ac.act_fep(obs_n, rnn_states, masks)

            next_obs_raw, rewards, dones, _ = env.step(actions)
            next_obs = sanitize_obs(extract_obs(next_obs_raw).to(device)).float()

            obs_rms.update(next_obs)
            next_obs_n = obs_rms.normalize(next_obs)

            rewards = torch.nan_to_num(rewards.to(device), nan=0.0, posinf=1e6, neginf=-1e6).float()
            masks = (~dones.bool()).float().unsqueeze(-1).to(device)

            # ✅ rollouts에는 'raw 정규화 관측'만 저장 (WM feature 금지)
            rollouts.insert(next_obs_n, rnn_states, actions, log_probs, values, rewards, masks)

            if render_enabled and (step % max(1, render_interval) == 0):
                try:
                    env.render(mode="human")
                except Exception:
                    pass

            obs = next_obs
            obs_n = next_obs_n

        # ── bootstrap value: WM feature 포함해 크리틱에 입력해야 함
        with torch.no_grad():
            x_val = ac._prep(obs_n)           # raw_norm -> [obs, z, z_next]
            next_value = ac.vf(x_val).squeeze(-1)

        rollouts.compute_returns(next_value, gamma, gae_lambda)

        logs = algo.update(rollouts)
        rollouts.after_update()

        if (update + 1) % log_interval == 0 and isinstance(logs, dict):
            pretty = {k: (float(v) if hasattr(v, "item") else v) for k, v in logs.items()}
            print(f"[Update {update+1}] {pretty}")

        # ── 간단 평가
        if (update + 1) % eval_interval == 0:
            with torch.no_grad():
                test_obs_raw = env.reset()
                test_obs = sanitize_obs(extract_obs(test_obs_raw).to(device)).float()
                test_obs_n = obs_rms.normalize(test_obs)

                test_ret = torch.zeros(num_envs, device=device)
                test_masks = torch.ones(num_envs, 1, device=device)
                test_rnn = torch.zeros(num_envs, 1, device=device)
                for _ in range(200):
                    v, a, _, _ = ac.act_fep(test_obs_n, test_rnn, test_masks)
                    test_obs_raw, r, d, _ = env.step(a)
                    test_obs = sanitize_obs(extract_obs(test_obs_raw).to(device)).float()
                    test_obs_n = obs_rms.normalize(test_obs)
                    test_ret += r.to(device)
                    test_masks = (~d.bool()).float().unsqueeze(-1).to(device)
                print(f"[Eval {update+1}] mean_return={test_ret.mean().item():.2f}")

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
