"""
Training script for Social-STGCNN.

Examples:
    python -m src.train_stgcnn --scene eth --epochs 250
    python -m src.train_stgcnn --scene all --epochs 250

IMPORTANT METHODOLOGICAL NOTE (must be stated in the report):
Social-STGCNN is a PROBABILISTIC model -- it predicts a distribution, not a single path.
In the literature its ADE/FDE is reported as "Best-of-20": 20 trajectories are sampled
and the best one is kept. The LSTM baseline is deterministic and produces one path.
Comparing those two numbers directly is incorrect (it pits 1 attempt against the best of
20), so this script computes BOTH variants:
    - ADE/FDE (deterministic): uses the mean of the distribution (mu) -- a fair
      comparison against the LSTM
    - ADE/FDE (best-of-20): the standard protocol from the papers -- for comparison
      against published figures
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_loader import TrajectoryDataset, seq_collate
from src.metrics import ErrorAccumulator, displacement_error, final_displacement_error, relative_to_abs
from src.models.social_stgcnn import SocialSTGCNN, bivariate_loss, build_adjacency

SCENES = ["eth", "hotel", "univ", "zara1", "zara2"]


def make_loader(data_dir: str, split: str, shuffle: bool, batch_size: int = 32):
    """A batch = several scenes packed into one block-diagonal graph (see build_adjacency)."""
    dset = TrajectoryDataset(os.path.join(data_dir, split))
    return dset, DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=seq_collate)


def prepare_inputs(obs_traj: torch.Tensor, obs_traj_rel: torch.Tensor, seq_start_end=None):
    """Turns (T, V, 2) into the inputs the model expects: v=(1, 2, T, V), A=(1, T, V, V)."""
    v = obs_traj_rel.permute(2, 0, 1).unsqueeze(0)              # (1, 2, T, V)
    a = build_adjacency(obs_traj, seq_start_end).unsqueeze(0)   # (1, T, V, V)
    return v, a


def sample_trajectories(pred_params: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Samples trajectories from the predicted bivariate Gaussian distribution.

    Args:
        pred_params: (T, V, 5)
        n_samples: number of samples (K)
    Returns:
        (K, T, V, 2) sampled relative displacements
    """
    mu = pred_params[..., 0:2]
    sx = torch.exp(pred_params[..., 2]).clamp(min=1e-3)
    sy = torch.exp(pred_params[..., 3]).clamp(min=1e-3)
    corr = torch.tanh(pred_params[..., 4]).clamp(min=-0.99, max=0.99)

    cov = torch.zeros(*mu.shape[:-1], 2, 2, device=mu.device)
    cov[..., 0, 0] = sx**2
    cov[..., 0, 1] = corr * sx * sy
    cov[..., 1, 0] = corr * sx * sy
    cov[..., 1, 1] = sy**2

    dist = torch.distributions.MultivariateNormal(mu, covariance_matrix=cov)
    return dist.sample((n_samples,))


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str, n_samples: int = 20):
    """Returns (ade_det, fde_det, ade_best20, fde_best20), all in metres."""
    model.eval()
    det = ErrorAccumulator()
    best_ade_sum, best_fde_sum, n_total = 0.0, 0.0, 0

    for obs, pred_gt, obs_rel, _, sse in loader:
        obs, pred_gt, obs_rel = obs.to(device), pred_gt.to(device), obs_rel.to(device)
        v, a = prepare_inputs(obs, obs_rel, sse)
        out = model(v, a).squeeze(0).permute(1, 2, 0)  # (T_pred, V, 5)

        # --- deterministic variant: the mean of the distribution ---
        pred_abs = relative_to_abs(out[..., 0:2], obs[-1])
        det.update(pred_abs, pred_gt)

        # --- best-of-K variant (vectorised over all K samples at once) ---
        samples = sample_trajectories(out, n_samples)          # (K, T, V, 2)
        abs_k = torch.cumsum(samples, dim=1) + obs[-1]         # (K, T, V, 2)
        err = torch.norm(abs_k - pred_gt.unsqueeze(0), dim=-1)  # (K, T, V)
        # the minimum is taken per agent (the standard protocol in the literature)
        best_ade_sum += err.mean(dim=1).min(dim=0).values.sum().item()
        best_fde_sum += err[:, -1].min(dim=0).values.sum().item()
        n_total += pred_gt.shape[1]

    return det.ade, det.fde, best_ade_sum / n_total, best_fde_sum / n_total


def train_one_scene(scene: str, args) -> dict:
    device = args.device
    torch.manual_seed(args.seed)

    data_dir = os.path.join(args.data_root, scene)
    _, train_loader = make_loader(data_dir, "train", True, args.batch_size)
    _, val_loader = make_loader(data_dir, "val", False, args.batch_size)
    _, test_loader = make_loader(data_dir, "test", False, args.batch_size)

    model = SocialSTGCNN(
        n_stgcnn=args.n_stgcnn,
        n_txpcnn=args.n_txpcnn,
        output_feat=args.output_feat,
        dropout=args.dropout,
        kernel_size=args.kernel_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.5)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f"stgcnn_{scene}.pt")

    best_val = float("inf")
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, nb = 0.0, 0

        for obs, _, obs_rel, pred_rel_gt, sse in train_loader:
            obs, obs_rel, pred_rel_gt = obs.to(device), obs_rel.to(device), pred_rel_gt.to(device)
            v, a = prepare_inputs(obs, obs_rel, sse)
            out = model(v, a).squeeze(0).permute(1, 2, 0)  # (T_pred, V, 5)

            optimizer.zero_grad()
            loss = bivariate_loss(out, pred_rel_gt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            total += loss.item()
            nb += 1

        scheduler.step()

        # validation is expensive (sampling), so it runs every val_every epochs
        if epoch % args.val_every == 0 or epoch == args.epochs or epoch == 1:
            val_det_ade, _, val_best_ade, _ = evaluate(
                model, val_loader, device, n_samples=args.val_samples
            )
            history.append(
                {"epoch": epoch, "loss": total / nb, "val_ade_det": val_det_ade, "val_ade_best": val_best_ade}
            )
            if val_best_ade < best_val:
                best_val = val_best_ade
                torch.save(model.state_dict(), ckpt_path)
            print(
                f"[{scene}] epoch {epoch:3d}/{args.epochs}  nll={total/nb:.4f}  "
                f"val_ADE(det)={val_det_ade:.3f}  val_ADE(best{args.val_samples})={val_best_ade:.3f}",
                flush=True,
            )

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    ade_det, fde_det, ade_20, fde_20 = evaluate(model, test_loader, device, n_samples=20)
    elapsed = time.time() - t0
    print(
        f"[{scene}] TEST  det: ADE={ade_det:.3f} FDE={fde_det:.3f} | "
        f"best-of-20: ADE={ade_20:.3f} FDE={fde_20:.3f}  ({elapsed:.0f}s)\n",
        flush=True,
    )

    result = {
        "scene": scene,
        "test_ade_det": ade_det,
        "test_fde_det": fde_det,
        "test_ade_best20": ade_20,
        "test_fde_best20": fde_20,
        "epochs": args.epochs,
        "seconds": elapsed,
        "history": history,
    }
    with open(os.path.join(args.out_dir, f"stgcnn_{scene}_log.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="eth", help="eth|hotel|univ|zara1|zara2|all")
    p.add_argument("--data_root", default="data")
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_step", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=32, help="scenes packed into one block-diagonal graph")
    p.add_argument("--val_every", type=int, default=5, help="run validation every N epochs")
    p.add_argument("--n_stgcnn", type=int, default=1)
    p.add_argument("--n_txpcnn", type=int, default=5)
    p.add_argument("--output_feat", type=int, default=5)
    p.add_argument("--kernel_size", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--val_samples", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    scenes = SCENES if args.scene == "all" else [args.scene]
    results = [train_one_scene(s, args) for s in scenes]

    if len(results) > 1:
        keys = ["test_ade_det", "test_fde_det", "test_ade_best20", "test_fde_best20"]
        avg = {k: sum(r[k] for r in results) / len(results) for k in keys}
        print("=" * 68)
        print(f"{'SCENA':<10}{'ADE det':>13}{'FDE det':>13}{'ADE b20':>13}{'FDE b20':>13}")
        print("-" * 68)
        for r in results:
            print(
                f"{r['scene']:<10}{r['test_ade_det']:>13.3f}{r['test_fde_det']:>13.3f}"
                f"{r['test_ade_best20']:>13.3f}{r['test_fde_best20']:>13.3f}"
            )
        print("-" * 68)
        print(
            f"{'PROSEK':<10}{avg['test_ade_det']:>13.3f}{avg['test_fde_det']:>13.3f}"
            f"{avg['test_ade_best20']:>13.3f}{avg['test_fde_best20']:>13.3f}"
        )
        print("=" * 68)
        with open(os.path.join(args.out_dir, "stgcnn_summary.json"), "w") as f:
            json.dump({"per_scene": results, "avg": avg}, f, indent=2)


if __name__ == "__main__":
    main()


