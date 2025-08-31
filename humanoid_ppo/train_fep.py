# train_fep.py
import argparse
import yaml
import numpy as np
from tqdm import trange

# ⚠️ 반드시 이 순서!
from isaacgym import gymapi      # 1) isaacgym(gymapi) 먼저
import isaacgymenvs              # 2) isaacgymenvs 다음
import torch                     # 3) torch는 가장 나중
from torch.utils.data import DataLoader, TensorDataset

from tasks.make_env import make_isaac_env, extract_obs, sanitize_obs
from ppo.agent import MLPActorCritic
from ppo.rollouts_fep import PPORollouts
from ppo.fep_ppo import FEP_PPO
from models.policy_prior import PolicyPrior


class RunningNorm:
    """관측 러닝 정규화 (per-dimension mean/var)."""
    def __init__(self, shape, eps=1e-5, device="cpu"):
        self.mean = torch.zeros(shape, device=device)
        self.var  = torch.ones(shape, device=device)
        self.count = torch.tensor(eps, device=device)

    @torch.no_grad()
    def update(self, x):
        # x: [N, D]
        batch_mean = x.mean(dim=0)
        batch_var  = x.var(dim=0, unbiased=False)
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

    def normalize(self, x):
        return (x - self.mean) / torch.sqrt(self.var + 1e-6)


def set_seed(s: int):
    torch.manual_seed(s)
    np.random.seed(s)


def maybe_make_demos(cfg):
    use = cfg.get("use_demos", False)
    path = cfg.get("demos_path", "")
    if not use or not path:
        return None

    if path.endswith(".pt"):
        data = torch.load(path, map_location="cpu")
        obs, act = data["obs"], data["act"]
    else:
        arr = np.load(path)
        obs, act = torch.tensor(arr["obs"]), torch.tensor(arr["act"])

    obs = obs.float()
    act = act.float()
    ds = TensorDataset(obs, act)
    return DataLoader(
        ds,
        batch_size=int(cfg.get("demos_batch_size", 256)),
        shuffle=True,
        drop_last=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg.get("seed", 1)))

    sim_device  = cfg.get("sim_device", None)   # "cuda:0" / "cpu" / None(자동)
    rl_device   = cfg.get("rl_device", None)
    headless    = bool(cfg.get("headless", True))
    graphics_id = int(cfg.get("graphics_device_id", 0))
    render_enabled = bool(cfg.get("render", False))
    render_interval = int(cfg.get("render_interval", 1))

    env, device, obs_dim, act_dim = make_isaac_env(
        task_name=cfg["task_name"],
        num_envs=int(cfg["num_envs"]),
        seed=int(cfg["seed"]),
        sim_device=sim_device,
        rl_device=rl_device,
        headless=headless,
        graphics_device_id=graphics_id,
    )

    # 카메라 시점(뷰어 있을 때 한 번만)
    viewer = getattr(env, "viewer", None)
    gym    = getattr(env, "gym", None)
    sim    = getattr(env, "sim", None)
    if (viewer is not None) and (gym is not None):
        eye = gymapi.Vec3(8.0, 8.0, 3.0)
        at  = gymapi.Vec3(0.0, 0.0, 1.0)
        gym.viewer_camera_look_at(viewer, None, eye, at)

    ac = MLPActorCritic(obs_dim, act_dim, hidden_size=int(cfg.get("hidden_size", 256))).to(device)
    prior = PolicyPrior(obs_dim, act_dim, hidden_size=int(cfg.get("hidden_size", 256))).to(device)
    demos_loader = maybe_make_demos(cfg)

    algo = FEP_PPO(
        actor_critic=ac,
        policy_prior=prior,
        prior_coef=float(cfg.get("prior_coef", 1.0)),
        prior_lr=float(cfg.get("prior_lr", 3e-4)),
        demos_loader=demos_loader,
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

    rollouts = PPORollouts(obs_dim, act_dim, horizon, num_envs, device)

    # 초기 관측
    obs_raw = env.reset()
    obs = extract_obs(obs_raw).to(device)
    obs = sanitize_obs(obs)
    # 러닝 정규화 준비
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
                values, actions, log_probs, rnn_states = ac.act_fep(obs_n, rnn_states, masks)

            next_obs_raw, rewards, dones, _ = env.step(actions)
            next_obs = extract_obs(next_obs_raw).to(device)
            next_obs = sanitize_obs(next_obs, clip=100.0)

            # 정규화 업데이트 → 적용
            obs_rms.update(next_obs)
            next_obs_n = obs_rms.normalize(next_obs)

            # 보상/마스크 세척
            rewards = torch.nan_to_num(rewards.to(device), nan=0.0, posinf=1e6, neginf=-1e6)
            masks = (~dones.bool()).float().unsqueeze(-1).to(device)

            # 롤아웃에는 정규화된 관측을 저장
            rollouts.insert(next_obs_n, rnn_states, actions, log_probs, values, rewards, masks)

            # 다음 스텝 입력 관측
            obs = next_obs
            obs_n = next_obs_n

            # --- 렌더링: 뷰어가 있고, render 켜져 있으면 N스텝마다 그리기 ---
            if render_enabled and (step % render_interval == 0):
                if (viewer is not None) and (gym is not None) and (sim is not None):
                    gym.step_graphics(sim)
                    gym.draw_viewer(viewer, sim, True)
                    gym.sync_frame_time(sim)

        with torch.no_grad():
            next_value = ac.vf(obs_n).squeeze(-1)
        rollouts.compute_returns(next_value, gamma, gae_lambda)

        logs = algo.update(rollouts)
        rollouts.after_update()

        if (update + 1) % log_interval == 0:
            print({k: round(float(v), 4) for k, v in logs.items()})

        if (update + 1) % eval_interval == 0:
            with torch.no_grad():
                test_obs_raw = env.reset()
                test_obs = extract_obs(test_obs_raw).to(device)
                test_obs = sanitize_obs(test_obs)
                test_obs_n = obs_rms.normalize(test_obs)

                test_ret = torch.zeros(num_envs, device=device)
                test_masks = torch.ones(num_envs, 1, device=device)
                test_rnn = torch.zeros(num_envs, 1, device=device)
                for _ in range(200):
                    v, a, _, _ = ac.act_fep(test_obs_n, test_rnn, test_masks)
                    test_obs_raw, r, d, _ = env.step(a)
                    test_obs = extract_obs(test_obs_raw).to(device)
                    test_obs = sanitize_obs(test_obs)
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
