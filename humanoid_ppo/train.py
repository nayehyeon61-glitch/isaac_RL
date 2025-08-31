import argparse
import yaml
import os
import torch
import numpy as np
from tqdm import trange

from tasks.make_env import make_isaac_env
from ppo.agent import MLPActorCritic
from ppo.storage import RolloutBuffer
from ppo.ppo import PPO

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    return p.parse_args()

def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    env, device, obs_dim, act_dim = make_isaac_env(
        cfg["task_name"], cfg["num_envs"], seed=cfg["seed"]
    )

    # 모델/알고리즘 초기화
    ac = MLPActorCritic(obs_dim, act_dim, hidden_size=cfg["hidden_size"]).to(device)
    algo = PPO(
        ac, lr=cfg["learning_rate"], clip_coef=cfg["clip_coef"], vf_coef=cfg["vf_coef"],
        ent_coef=cfg["ent_coef"], max_grad_norm=cfg["max_grad_norm"],
        update_epochs=cfg["update_epochs"], mini_batch_size=cfg["mini_batch_size"]
    )

    horizon = cfg["horizon"]
    num_envs = cfg["num_envs"]
    buffer = RolloutBuffer(obs_dim, act_dim, horizon, num_envs, device)

    obs = env.reset().to(device)  # IsaacGymEnvs는 torch tensor 반환
    global_step = 0

    for update in trange(cfg["total_updates"], desc="PPO Updates"):
        buffer.ptr = 0
        for t in range(horizon):
            with torch.no_grad():
                action, logp, value = ac.act(obs)
            # Isaac Gym 연속액션은 [-1,1] 범위의 텐서를 기대 (이미 tanh-squash되어 있음)
            next_obs, reward, done, info = env.step(action)
            # env가 반환하는 텐서가 같은 device에 있는지 확실히
            next_obs = next_obs.to(device)
            reward = reward.to(device)
            done = done.to(device, dtype=torch.bool)

            buffer.add(obs, action, logp, reward, done, value)
            obs = next_obs
            global_step += num_envs

        # 마지막 상태의 value
        with torch.no_grad():
            last_value = ac.vf(obs).squeeze(-1)
        buffer.add_last_value(last_value, obs)

        # GAE 계산 및 PPO 업데이트
        b_obs, b_act, b_logp, b_ret, b_adv, b_val = buffer.compute_gae(
            cfg["gamma"], cfg["gae_lambda"]
        )
        info = algo.update(b_obs, b_act, b_logp, b_ret, b_adv, b_val)

        if (update + 1) % cfg["log_interval"] == 0:
            ep_rew = None
            if isinstance(info, dict):
                # 간단 로그
                print(f"[Upd {update+1}] "
                      f"pi={info['policy_loss']:.3f} "
                      f"vf={info['value_loss']:.3f} "
                      f"ent={info['entropy']:.3f}")

        # 간단한 평가(옵션): 학습 안정 후 주기적으로 수행 가능
        if (update + 1) % cfg["eval_interval"] == 0:
            with torch.no_grad():
                test_obs = env.reset().to(device)
                ep_ret = torch.zeros(num_envs, device=device)
                for _ in range(200):  # 짧게 rollout
                    a, _, _ = ac.act(test_obs)
                    test_obs, r, d, _ = env.step(a)
                    ep_ret += r.to(device)
                print(f"[Eval @Upd {update+1}] mean_return={ep_ret.mean().item():.2f}")

    # 종료
    try:
        env.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()

