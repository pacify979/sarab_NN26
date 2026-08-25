"""
Training script for the Vanilla LSTM baseline.

Examples:
    python -m src.train_lstm --scene eth --epochs 100
    python -m src.train_lstm --scene all --epochs 100                  # all five scenes
    python -m src.train_lstm --scene all --epochs 250 --probabilistic  # control model

Leave-one-scene-out protocol: for the scene `eth`, the folder data/eth/train contains the
data from the remaining four scenes, while data/eth/test contains only eth. The model is
therefore tested on a scene it has never seen.
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn

from src.data_loader import data_loader
from src.metrics import ErrorAccumulator, relative_to_abs
from src.models.lstm_baseline import VanillaLSTM
from src.models.social_stgcnn import bivariate_loss

SCENES = ["eth", "hotel", "univ", "zara1", "zara2"]


def l2_loss(pred_rel: torch.Tensor, gt_rel: torch.Tensor) -> torch.Tensor:
    """Mean Euclidean error over displacements -- directly correlated with ADE."""
    return torch.norm(pred_rel - gt_rel, p=2, dim=-1).mean()


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str):
    """Computes ADE/FDE in METRES over absolute coordinates (deterministic)."""
    model.eval()
    acc = ErrorAccumulator()
    for obs, pred_gt, obs_rel, _, _ in loader:
        obs, pred_gt, obs_rel = obs.to(device), pred_gt.to(device), obs_rel.to(device)
        pred_rel = model(obs_rel)[..., :2]
        # displacements -> absolute positions, starting from the last observed point
        pred_abs = relative_to_abs(pred_rel, obs[-1])
        acc.update(pred_abs, pred_gt)
    return acc.ade, acc.fde


@torch.no_grad()
def evaluate_prob(model: nn.Module, loader, device: str, n_samples: int = 20):
    """Evaluates the probabilistic variant: returns (ade_det, fde_det, ade_bestK, fde_bestK).

    Identical protocol to the ST-GCNN one, so that the figures are directly comparable.
    """
    from src.train_stgcnn import sample_trajectories

    model.eval()
    det = ErrorAccumulator()
    best_ade_sum, best_fde_sum, n_total = 0.0, 0.0, 0

    for obs, pred_gt, obs_rel, _, _ in loader:
        obs, pred_gt, obs_rel = obs.to(device), pred_gt.to(device), obs_rel.to(device)
        out = model(obs_rel)  # (T_pred, N, 5)

        det.update(relative_to_abs(out[..., :2], obs[-1]), pred_gt)

        samples = sample_trajectories(out, n_samples)          # (K, T, N, 2)
        abs_k = torch.cumsum(samples, dim=1) + obs[-1]
        err = torch.norm(abs_k - pred_gt.unsqueeze(0), dim=-1)  # (K, T, N)
        best_ade_sum += err.mean(dim=1).min(dim=0).values.sum().item()
        best_fde_sum += err[:, -1].min(dim=0).values.sum().item()
        n_total += pred_gt.shape[1]

    return det.ade, det.fde, best_ade_sum / n_total, best_fde_sum / n_total


def train_one_scene(scene: str, args) -> dict:
    device = args.device
    torch.manual_seed(args.seed)

    data_dir = os.path.join(args.data_root, scene)
    _, train_loader = data_loader(data_dir, "train", batch_size=args.batch_size, shuffle=True)
    _, val_loader = data_loader(data_dir, "val", batch_size=args.batch_size, shuffle=False)
    _, test_loader = data_loader(data_dir, "test", batch_size=args.batch_size, shuffle=False)

    model = VanillaLSTM(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        probabilistic=args.probabilistic,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)
    prefix = "lstm_prob" if args.probabilistic else "lstm"
    ckpt_path = os.path.join(args.out_dir, f"{prefix}_{scene}.pt")

    best_val_ade = float("inf")
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, nb = 0.0, 0
        for obs, _, obs_rel, pred_rel_gt, _ in train_loader:
            obs_rel, pred_rel_gt = obs_rel.to(device), pred_rel_gt.to(device)
            optimizer.zero_grad()
            out = model(obs_rel)
            # the probabilistic variant learns a distribution (the same NLL as ST-GCNN),
            # the deterministic one learns a single point (L2)
            loss = bivariate_loss(out, pred_rel_gt) if args.probabilistic else l2_loss(out, pred_rel_gt)
            loss.backward()
            # gradient clipping: the LSTM decoder is autoregressive, so gradients can explode
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            total += loss.item()
            nb += 1

        if args.probabilistic:
            # selection by best-of-K ADE, the same criterion used for ST-GCNN
            _, _, val_ade, val_fde = evaluate_prob(model, val_loader, device, args.val_samples)
        else:
            val_ade, val_fde = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "loss": total / nb, "val_ade": val_ade, "val_fde": val_fde})

        # the checkpoint is selected by validation ADE, never by training loss
        if val_ade < best_val_ade:
            best_val_ade = val_ade
            torch.save(model.state_dict(), ckpt_path)

        if epoch % args.log_every == 0 or epoch == 1:
            print(
                f"[{scene}] epoch {epoch:3d}/{args.epochs}  "
                f"loss={total/nb:.4f}  val_ADE={val_ade:.3f}  val_FDE={val_fde:.3f}"
            )

    # final evaluation using the best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    elapsed = time.time() - t0

    result = {"scene": scene, "epochs": args.epochs, "seconds": elapsed, "history": history}
    if args.probabilistic:
        ade_det, fde_det, ade_20, fde_20 = evaluate_prob(model, test_loader, device, 20)
        print(
            f"[{scene}] TEST  det: ADE={ade_det:.3f} FDE={fde_det:.3f} | "
            f"best-of-20: ADE={ade_20:.3f} FDE={fde_20:.3f}  ({elapsed:.0f}s)\n"
        )
        result.update({
            "test_ade_det": ade_det, "test_fde_det": fde_det,
            "test_ade_best20": ade_20, "test_fde_best20": fde_20,
        })
    else:
        test_ade, test_fde = evaluate(model, test_loader, device)
        print(f"[{scene}] TEST  ADE={test_ade:.3f} m  FDE={test_fde:.3f} m  ({elapsed:.0f}s)\n")
        result.update({"test_ade": test_ade, "test_fde": test_fde, "best_val_ade": best_val_ade})

    with open(os.path.join(args.out_dir, f"{prefix}_{scene}_log.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="eth", help="eth|hotel|univ|zara1|zara2|all")
    p.add_argument("--data_root", default="data")
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--embedding_dim", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--probabilistic", action="store_true",
                   help="predict a distribution (bivariate NLL) instead of a single point -- "
                        "enables a fair best-of-20 comparison against ST-GCNN")
    p.add_argument("--val_samples", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    scenes = SCENES if args.scene == "all" else [args.scene]
    results = [train_one_scene(s, args) for s in scenes]

    if len(results) > 1:
        keys = (["test_ade_det", "test_fde_det", "test_ade_best20", "test_fde_best20"]
                if args.probabilistic else ["test_ade", "test_fde"])
        avg = {k: sum(r[k] for r in results) / len(results) for k in keys}
        w = 13
        print("=" * (10 + w * len(keys)))
        print(f"{'SCENE':<10}" + "".join(f"{k.replace('test_', ''):>{w}}" for k in keys))
        print("-" * (10 + w * len(keys)))
        for r in results:
            print(f"{r['scene']:<10}" + "".join(f"{r[k]:>{w}.3f}" for k in keys))
        print("-" * (10 + w * len(keys)))
        print(f"{'AVERAGE':<10}" + "".join(f"{avg[k]:>{w}.3f}" for k in keys))
        print("=" * (10 + w * len(keys)))

        prefix = "lstm_prob" if args.probabilistic else "lstm"
        with open(os.path.join(args.out_dir, f"{prefix}_summary.json"), "w") as f:
            json.dump({"per_scene": results, "avg": avg}, f, indent=2)


if __name__ == "__main__":
    main()
