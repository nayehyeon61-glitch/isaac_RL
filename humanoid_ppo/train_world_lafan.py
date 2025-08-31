# train_world_lafan.py
# LaFAN DOF sequences loader + WorldModel(Diffusion + Neural ODE) trainer
# 폴더 구조 예:
#   ./Resource/
#       walk/*.dof or *.bvh.dof or *.npy
#       run/*.dof  or *.bvh.dof  or *.npy
#
# 사용 예:
#   python train_world_lafan.py --data_root ./Resource --dof_dim 35 --seq_len 64

import os
import glob
import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ── 학습할 WorldModel (미리 제공한 model/model.py 사용) ──
# 패키지 경로가 맞지 않으면, 아래 3줄 주석 해제해서 루트 경로를 sys.path에 추가하세요.
# import sys
# ROOT = Path(__file__).resolve().parent
# sys.path.append(str(ROOT))
from model.model import WorldModel


# ------------------------------
# 경로/파일 수집 유틸
# ------------------------------
def collect_dof_files(
    data_root: str,
    subdirs=("walk", "run"),
    patterns=("*.dof", "*.bvh.dof", "*.npy"),
    recursive=True,
) -> List[str]:
    files = []
    for sd in subdirs:
        base = os.path.join(data_root, sd)
        for pat in patterns:
            if recursive:
                files.extend(glob.glob(os.path.join(base, "**", pat), recursive=True))
            else:
                files.extend(glob.glob(os.path.join(base, pat)))
    # 중복 제거 + 정렬
    files = sorted(list(dict.fromkeys(files)))
    return files


# ------------------------------
# 안전한 DOF 텍스트 로더
# ------------------------------
def read_dof_txt(path: str, dof_dim: int) -> np.ndarray:
    """
    안전 로더:
    - 인코딩 문제 무시(errors='replace'), 쉼표/탭/연속 공백 정리
    - 숫자만 추출(비숫자 토큰은 무시)
    - 각 줄을 dof_dim에 맞춰 trim/pad
    - 완전 빈 줄은 스킵
    반환: [T, dof_dim] float32
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        txt = f.read()

    txt = txt.replace(",", " ").replace("\t", " ")
    lines = [ln.strip() for ln in txt.splitlines()]

    rows = []
    bad_lines = 0
    for ln in lines:
        if not ln:
            continue
        parts = [p for p in ln.split(" ") if p]  # 연속 공백 제거
        nums = []
        for p in parts:
            try:
                p2 = p.replace("\xa0", "").strip()
                if p2 == "":
                    continue
                nums.append(float(p2))
            except Exception:
                # 숫자 아닌 토큰은 무시
                continue

        if len(nums) == 0:
            bad_lines += 1
            continue

        if len(nums) >= dof_dim:
            nums = nums[:dof_dim]
        else:
            nums = nums + [0.0] * (dof_dim - len(nums))

        rows.append(nums)

    if len(rows) == 0:
        return np.zeros((0, dof_dim), dtype=np.float32)

    arr = np.asarray(rows, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if bad_lines > 0:
        print(f"[read_dof_txt] {os.path.basename(path)}: skipped {bad_lines} empty/non-numeric lines")

    return arr  # [T, dof_dim]


# ------------------------------
# Dataset
# ------------------------------
class LaFANDOFDataset(Dataset):
    """
    ./Resource/walk/*.dof, ./Resource/run/*.dof 등을 모아
    길이 seq_len+1 의 서브시퀀스를 샘플링해 반환.
    반환 텐서: [T+1, D] (T = seq_len)
    """
    def __init__(
        self,
        data_root: str,
        dof_dim: int = 35,
        seq_len: int = 64,
        split: str = "train",
        train_ratio: float = 0.9,
        cache: bool = True,
        recursive: bool = True,
        stride: int = 1,  # 슬라이딩 간격
    ):
        super().__init__()
        self.data_root = data_root
        self.dof_dim = dof_dim
        self.seq_len = seq_len
        self.split = split
        self.train_ratio = train_ratio
        self.cache = cache
        self.recursive = recursive
        self.stride = max(1, int(stride))

        # 1) 파일 수집
        self.files = collect_dof_files(
            data_root=data_root,
            subdirs=("walk", "run"),
            patterns=("*.dof", "*.bvh.dof", "*.npy"),
            recursive=recursive,
        )
        if len(self.files) == 0:
            raise FileNotFoundError(
                f"DOF 파일을 찾지 못했습니다. data_root={data_root}\n"
                f" - 예: {os.path.join(data_root, 'walk', '**', '*.bvh.dof')}"
            )

        # 2) 파일 파싱 → 길이가 충분한 것만 keep (T >= seq_len+1)
        seqs_all = []
        lens_all = []
        kept = dropped_short = dropped_empty = 0

        for fp in self.files:
            if fp.lower().endswith(".npy"):
                arr = np.load(fp).astype(np.float32)
            else:
                arr = read_dof_txt(fp, dof_dim=self.dof_dim)

            if arr.size == 0 or arr.shape[0] == 0:
                dropped_empty += 1
                continue

            T = int(arr.shape[0])
            if T < (self.seq_len + 1):
                dropped_short += 1
                continue

            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

            seqs_all.append(arr)
            lens_all.append(T)
            kept += 1

        if kept == 0:
            msg = (
                "적절한 길이의 DOF 시퀀스가 없습니다.\n"
                f"- data_root={data_root}\n"
                f"- files={len(self.files)}개 찾음, empty={dropped_empty}, short(<{self.seq_len+1})={dropped_short}\n"
                "→ 해결책: seq_len을 줄이거나 데이터 파싱을 점검하세요."
            )
            raise RuntimeError(msg)

        # 3) train/val split (파일 단위)
        idxs = list(range(len(seqs_all)))
        random.shuffle(idxs)
        n_train = int(len(idxs) * self.train_ratio)
        keep_idx = idxs[:n_train] if split == "train" else idxs[n_train:]

        self.seqs: List[np.ndarray] = [seqs_all[i] for i in keep_idx]
        self.lengths: List[int] = [lens_all[i] for i in keep_idx]

        total_len = sum(self.lengths)
        print(f"[LaFANDOFDataset:{split}] files={len(self.seqs)}  total_frames={total_len}  "
              f"minT={min(self.lengths)}  maxT={max(self.lengths)}  seq_len={self.seq_len}  stride={self.stride}")

        # 4) 서브시퀀스 인덱스 전개
        self.index: List[Tuple[int, int]] = []
        for i, T in enumerate(self.lengths):
            max_start = T - (self.seq_len + 1)
            if max_start < 0:
                continue
            for s in range(0, max_start + 1, self.stride):
                self.index.append((i, s))

        if len(self.index) == 0:
            raise RuntimeError("유효한 서브시퀀스를 만들 수 없습니다. seq_len을 더 줄여보세요.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        i, s = self.index[idx]
        arr = self.seqs[i]
        chunk = arr[s : s + self.seq_len + 1]          # [T+1, D]
        return torch.from_numpy(chunk).float()         # [T+1, D]


# ------------------------------
# 학습 루프
# ------------------------------
def train_one_epoch(
    wm: WorldModel,
    loader: DataLoader,
    opt: optim.Optimizer,
    device: torch.device,
    log_every: int = 100,
):
    wm.train()
    step = 0
    avg_diff = 0.0
    avg_ode  = 0.0
    for batch in loader:
        # batch: [B, T+1, D]
        batch = batch.to(device)
        B, Tp1, D = batch.shape
        T = Tp1 - 1

        # (1) 확산 재구성: (B*T, D) 샘플링 → t는 랜덤
        x = batch[:, :-1, :].reshape(B * T, D)
        t = torch.randint(0, wm.diffusion.timesteps, (x.size(0),), device=device, dtype=torch.long)
        loss_diff = wm.recon_loss(x, t)

        # (2) ODE 잠재 1-step 예측
        loss_ode = wm.latent_roll_loss(batch)

        loss = loss_diff + loss_ode

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
        opt.step()

        avg_diff += loss_diff.item()
        avg_ode  += loss_ode.item()
        step += 1

        if step % log_every == 0:
            print(f"  step {step:6d} | loss_diff={avg_diff/step:.4f} | loss_ode={avg_ode/step:.4f}")

    return {"loss_diff": avg_diff / max(step, 1), "loss_ode": avg_ode / max(step, 1)}


@torch.no_grad()
def preview_rollout(wm: WorldModel, batch: torch.Tensor, device, horizon=10, steps=50):
    wm.eval()
    x0 = batch[0, 0, :].unsqueeze(0).to(device)    # [1, D]
    sample = wm.rollout(x0, horizon=horizon, steps=steps)  # [1, H, D]
    print(f"  rollout preview: mean={sample.mean().item():.4f} std={sample.std().item():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="폴더 루트 (예: ./Resource)")
    ap.add_argument("--dof_dim", type=int, default=35)
    ap.add_argument("--z_dim", type=int, default=64)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--stride", type=int, default=1, help="슬라이딩 윈도우 간격")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--train_ratio", type=float, default=0.9)
    ap.add_argument("--log_every", type=int, default=100)
    args = ap.parse_args()

    # ── data_root 경로 보정: 스크립트와 같은 레벨의 Resource를 자동 탐지 ──
    script_dir = Path(__file__).resolve().parent
    data_root_arg = Path(args.data_root)
    if not data_root_arg.is_absolute():
        data_root = (script_dir / data_root_arg).resolve()
    else:
        data_root = data_root_arg
    if not data_root.exists():
        candidate = (script_dir / "Resource").resolve()
        if candidate.exists():
            print(f"[info] {data_root} 가 없어 {candidate} 로 자동 보정합니다.")
            data_root = candidate
    print(f"[info] data_root={data_root}")
    args.data_root = str(data_root)

    device = torch.device(args.device)

    # Dataset / DataLoader
    ds_train = LaFANDOFDataset(
        data_root=args.data_root,
        dof_dim=args.dof_dim,
        seq_len=args.seq_len,
        split="train",
        train_ratio=args.train_ratio,
        stride=args.stride,
        recursive=True,
    )
    ds_val = LaFANDOFDataset(
        data_root=args.data_root,
        dof_dim=args.dof_dim,
        seq_len=args.seq_len,
        split="val",
        train_ratio=args.train_ratio,
        stride=args.stride,
        recursive=True,
    )
    train_loader = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
        drop_last=False,
    )

    # World Model
    wm = WorldModel(dof_dim=args.dof_dim, z_dim=args.z_dim, diff_steps=1000).to(device)
    opt = optim.Adam(wm.parameters(), lr=args.lr)

    # 학습
    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}/{args.epochs}]")
        logs = train_one_epoch(wm, train_loader, opt, device, log_every=args.log_every)

        # 간단 검증(롤아웃 미리보기 1회)
        for val_batch in val_loader:
            preview_rollout(wm, val_batch, device, horizon=10, steps=50)
            break

        print(f"  train: loss_diff={logs['loss_diff']:.4f} | loss_ode={logs['loss_ode']:.4f}")

    torch.save(wm.state_dict(), "world_model_diffode.pt")
    print("Saved model to world_model_diffode.pt")


if __name__ == "__main__":
    main()
