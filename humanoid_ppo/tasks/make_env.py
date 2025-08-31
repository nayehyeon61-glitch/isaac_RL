# tasks/make_env.py
# ⚠️ import 순서 절대 중요: gymapi → isaacgymenvs → torch
from isaacgym import gymapi
import isaacgymenvs
import torch


def extract_obs(obs):
    """IsaacGymEnvs가 dict 관측을 반환하는 버전 대응."""
    if torch.is_tensor(obs):
        return obs
    if isinstance(obs, dict):
        for k in ("obs", "policy", "obs_buf", "state", "features"):
            if k in obs and torch.is_tensor(obs[k]):
                return obs[k]
        for v in obs.values():
            if torch.is_tensor(v):
                return v
    raise TypeError(f"Unsupported observation type: {type(obs)}")


def sanitize_obs(x, clip=100.0):
    """관측값 세척: dtype 강제 + NaN/Inf 대체 + 클리핑."""
    if not torch.is_tensor(x):
        raise TypeError("sanitize_obs expects torch.Tensor")
    x = x.float()
    x = torch.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip)
    return torch.clamp(x, -clip, clip)


def make_isaac_env(
    task_name: str,
    num_envs: int,
    seed: int = 1,
    sim_device: str = None,
    rl_device: str = None,
    headless: bool = True,
    graphics_device_id: int = 0,
):
    """
    IsaacGymEnvs VecTask 생성 헬퍼
    - 최신 make 시그니처 대응: sim_device, rl_device, graphics_device_id 필요
    - import 순서: gymapi → isaacgymenvs → torch
    """
    if sim_device is None:
        sim_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if rl_device is None:
        rl_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env = isaacgymenvs.make(
        seed=seed,
        task=task_name,
        num_envs=num_envs,
        sim_device=sim_device,
        rl_device=rl_device,
        graphics_device_id=graphics_device_id,
        headless=headless,
    )

    device = torch.device(rl_device)

    obs_raw = env.reset()
    obs = extract_obs(obs_raw).to(device)
    obs = sanitize_obs(obs)
    obs_dim = int(obs.shape[-1])
    act_dim = int(env.action_space.shape[0])
    return env, device, obs_dim, act_dim
