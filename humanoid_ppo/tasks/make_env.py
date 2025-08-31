# tasks/make_env.py
from isaacgym import gymapi
import isaacgymenvs
import torch

def extract_obs(obs):
    if torch.is_tensor(obs):
        return obs
    if isinstance(obs, dict):
        if "obs" in obs and torch.is_tensor(obs["obs"]):
            return obs["obs"]
        for k in ("policy", "obs_buf", "state", "features"):
            if k in obs and torch.is_tensor(obs[k]):
                return obs[k]
        for v in obs.values():
            if torch.is_tensor(v):
                return v
    raise TypeError(f"Unsupported observation type: {type(obs)} (keys={list(obs.keys()) if isinstance(obs, dict) else None})")

def sanitize_obs(x, clip=100.0):
    x = torch.as_tensor(x).float()
    x = torch.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip)
    return torch.clamp(x, -clip, clip)

def make_isaac_env(task_name, num_envs, seed=1,
                   sim_device=None, rl_device=None,
                   headless=True, graphics_device_id=0):
    if sim_device is None:
        sim_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if rl_device is None:
        rl_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        env = isaacgymenvs.make(
            seed=seed,
            task=task_name,
            num_envs=num_envs,
            sim_device=sim_device,
            rl_device=rl_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
        )
    except Exception as e:
        raise RuntimeError(f"isaacgymenvs.make 실패: {e}")
    device = torch.device(rl_device)
    try:
        obs_raw = env.reset()
        obs_t = extract_obs(obs_raw).to(device)
        obs_t = sanitize_obs(obs_t)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        obs_dim = int(obs_t.shape[-1])
        if hasattr(env, "action_space") and getattr(env.action_space, "shape", None) is not None:
            act_dim = int(env.action_space.shape[0])
        elif hasattr(env, "num_actions"):
            act_dim = int(env.num_actions)
        else:
            act_dim = 21
    except Exception as e:
        raise RuntimeError(f"reset/obs 처리 실패: {e}")
    return env, device, obs_dim, act_dim
# 파일: tasks/make_env.py (맨 아래쪽에 추가)

def _vec3(x, y, z):
    from isaacgym import gymapi
    return gymapi.Vec3(float(x), float(y), float(z))

def set_viewer_camera_close(env,
                            env_index: int = 0,
                            distance: float = 3.0,     # 타겟에서 카메라까지 거리 (m)
                            height: float = 1.2,       # 카메라 높이 보정 (m)
                            yaw_deg: float = 30.0):    # 수평 회전 각 (deg, +는 반시계)
    """
    Isaac Gym viewer 카메라를 env_index의 첫 번째 액터(보통 휴머노이드 루트)로 당겨 맞춥니다.
    - viewer가 켜져 있어야 함(headless=False).
    - target은 (가능하면) 루트 포지션 사용, 없으면 (0,0,0) 근처로 추정.
    """
    gym = env.gym
    viewer = getattr(env, "viewer", None)
    if viewer is None:
        return  # headless 모드

    try:
        env_ptr = env.envs[env_index]
    except Exception:
        return

    # 타겟 위치 추출: 루트 상태 텐서가 있으면 거기서, 아니면 (0,0,0)
    target = None
    try:
        # isaacgymenvs VecTask는 보통 actor_root_state_tensor 보유
        root = env.actor_root_state_tensor  # shape [num_actors_total, 13] or [N*actors, 13]
        # 첫 번째 환경의 첫 액터가 0번이라고 가정 (대부분의 기본 태스크에서 맞습니다)
        tx, ty, tz = float(root[0, 0]), float(root[0, 1]), float(root[0, 2])
        target = (tx, ty, tz)
    except Exception:
        # fallback: 고정 원점 근처
        target = (0.0, 0.0, 1.0)

    # target 위로 살짝 올림
    tx, ty, tz = target[0], target[1], target[2] + 0.9

    # 방위각으로 오프셋 벡터 생성
    import math
    yaw = math.radians(yaw_deg)
    dx = distance * math.cos(yaw)
    dy = distance * math.sin(yaw)
    eye = _vec3(tx + dx, ty + dy, tz + height)
    tgt = _vec3(tx, ty, tz)

    # 뷰어 카메라 이동
    gym.viewer_camera_look_at(viewer, env_ptr, eye, tgt)


def follow_target_each_step(env,
                            env_index: int = 0,
                            distance: float = 3.0,
                            height: float = 1.2,
                            yaw_deg: float = 30.0):
    """
    스텝 루프에서 호출해 카메라가 타겟을 계속 따라가도록 함.
    """
    set_viewer_camera_close(env, env_index, distance, height, yaw_deg)
