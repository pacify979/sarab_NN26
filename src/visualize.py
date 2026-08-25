"""
Qualitative analysis: visualisation of predicted trajectories and probability heatmaps.

Examples:
    python -m src.visualize --scene zara1 --n_scenes 6
    python -m src.visualize --scene eth --n_scenes 4 --min_peds 4
    python -m src.visualize --scene zara1 --lang en --no_title --fig_dir docs/paper/figures/en

Produces two kinds of figure:
  1. trajectories_<scene>_<i>.png -- observed history, ground truth, LSTM and ST-GCNN
     predictions overlaid
  2. heatmap_<scene>_<i>.png      -- probability density of future motion (ST-GCNN),
     obtained by sampling from the learned bivariate distribution. This is the visual
     evidence that the model learns UNCERTAINTY: the density is narrow when the motion is
     predictable, and widens and bends around other pedestrians where personal space is
     respected.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from src.metrics import relative_to_abs
from src.models.lstm_baseline import VanillaLSTM
from src.models.social_stgcnn import SocialSTGCNN, build_adjacency
from src.train_stgcnn import make_loader, prepare_inputs, sample_trajectories

# palette: contrasty enough to survive greyscale printing
C_OBS = "#2b2b2b"
C_GT = "#1a9850"
C_LSTM = "#e08214"
C_STGCNN = "#3b6fd4"

# Figure labels. For the IEEE paper the figures are generated with --lang en and
# --no_title, because the IEEE convention puts the description in the caption rather than
# inside the figure itself.
LABELS = {
    "sr": {
        "obs": "Osmotreno (8 frejmova)",
        "gt": "Ground truth (12 frejmova)",
        "lstm": "LSTM baseline",
        "stgcnn": "Social-STGCNN",
        "obs_short": "Osmotreno",
        "gt_short": "Ground truth",
        "density": "relativna gustina verovatnoce",
        "traj_title": "Scena {scene} -- {n} pesaka: predikcija vs. ground truth",
        "heat_title": "Scena {scene} -- gustina verovatnoce kretanja (Social-STGCNN)",
        "loss_title": "Funkcija gubitka ({scene})",
        "ade_title": "Validacioni ADE ({scene})",
        "epoch": "epoha",
    },
    "en": {
        "obs": "Observed (8 frames)",
        "gt": "Ground truth (12 frames)",
        "lstm": "LSTM baseline",
        "stgcnn": "Social-STGCNN",
        "obs_short": "Observed",
        "gt_short": "Ground truth",
        "density": "relative probability density",
        "traj_title": "{scene} -- {n} pedestrians: prediction vs. ground truth",
        "heat_title": "{scene} -- predicted motion density (Social-STGCNN)",
        "loss_title": "Training loss ({scene})",
        "ade_title": "Validation ADE ({scene})",
        "epoch": "epoch",
    },
}


def load_models(scene: str, out_dir: str, device: str):
    lstm = VanillaLSTM().to(device)
    lstm.load_state_dict(
        torch.load(os.path.join(out_dir, f"lstm_{scene}.pt"), map_location=device, weights_only=True)
    )
    lstm.eval()

    stgcnn = SocialSTGCNN().to(device)
    stgcnn.load_state_dict(
        torch.load(os.path.join(out_dir, f"stgcnn_{scene}.pt"), map_location=device, weights_only=True)
    )
    stgcnn.eval()
    return lstm, stgcnn


@torch.no_grad()
def predict_both(lstm, stgcnn, obs, obs_rel, device):
    """Returns (lstm_abs, stgcnn_abs, stgcnn_params) for a single scene."""
    lstm_abs = relative_to_abs(lstm(obs_rel), obs[-1])

    v, a = prepare_inputs(obs, obs_rel)
    params = stgcnn(v, a).squeeze(0).permute(1, 2, 0)  # (T_pred, V, 5)
    stgcnn_abs = relative_to_abs(params[..., 0:2], obs[-1])
    return lstm_abs, stgcnn_abs, params


def plot_trajectories(obs, gt, lstm_pred, stgcnn_pred, path: str, title: str, L=None):
    """Compares the predictions of every agent in one scene."""
    L = L or LABELS["sr"]
    obs, gt = obs.cpu().numpy(), gt.cpu().numpy()
    lstm_pred, stgcnn_pred = lstm_pred.cpu().numpy(), stgcnn_pred.cpu().numpy()
    n_ped = obs.shape[1]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for i in range(n_ped):
        # history: solid line with a marker at the start
        ax.plot(obs[:, i, 0], obs[:, i, 1], "-", color=C_OBS, lw=2, alpha=0.85,
                label=L["obs"] if i == 0 else None)
        ax.plot(obs[0, i, 0], obs[0, i, 1], "o", color=C_OBS, ms=5)

        # join the last observed point to each prediction so the lines do not float
        def seg(pred):
            return np.concatenate([obs[-1:, i, :], pred[:, i, :]], axis=0)

        ax.plot(*seg(gt).T, "--", color=C_GT, lw=2.2,
                label=L["gt"] if i == 0 else None)
        ax.plot(*seg(lstm_pred).T, "-", color=C_LSTM, lw=1.8, alpha=0.9,
                label=L["lstm"] if i == 0 else None)
        ax.plot(*seg(stgcnn_pred).T, "-", color=C_STGCNN, lw=1.8, alpha=0.9,
                label=L["stgcnn"] if i == 0 else None)

        ax.plot(gt[-1, i, 0], gt[-1, i, 1], "*", color=C_GT, ms=12)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25, ls=":")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_heatmap(obs, gt, params, path: str, title: str, L=None, n_samples: int = 2000, bins: int = 140):
    """Heatmap of future-motion density, obtained by sampling the learned distribution.

    A large number of trajectories is sampled from the bivariate Gaussian the model
    predicted, and a 2D histogram of their positions is built. The result is a map of
    "where pedestrians will probably be over the next 4.8 seconds" -- exactly what a robot
    path planner needs in order to avoid socially unacceptable zones.
    """
    L = L or LABELS["sr"]
    samples = sample_trajectories(params, n_samples)          # (K, T, V, 2)
    abs_samples = torch.cumsum(samples, dim=1) + obs[-1]      # (K, T, V, 2)
    pts = abs_samples.reshape(-1, 2).cpu().numpy()

    obs_np, gt_np = obs.cpu().numpy(), gt.cpu().numpy()

    # the view bounds cover both the history and the samples
    all_x = np.concatenate([pts[:, 0], obs_np[..., 0].ravel(), gt_np[..., 0].ravel()])
    all_y = np.concatenate([pts[:, 1], obs_np[..., 1].ravel(), gt_np[..., 1].ravel()])
    pad = 1.0
    rng = [[all_x.min() - pad, all_x.max() + pad], [all_y.min() - pad, all_y.max() + pad]]

    H, xe, ye = np.histogram2d(pts[:, 0], pts[:, 1], bins=bins, range=rng)
    H = gaussian_filter(H, sigma=2.0)  # light smoothing so the histogram is not grainy

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(
        H.T, origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]],
        cmap="magma", aspect="auto", alpha=0.92,
    )
    fig.colorbar(im, ax=ax, label=L["density"])

    n_ped = obs_np.shape[1]
    for i in range(n_ped):
        ax.plot(obs_np[:, i, 0], obs_np[:, i, 1], "-", color="white", lw=2.2,
                label=L["obs_short"] if i == 0 else None)
        seg = np.concatenate([obs_np[-1:, i, :], gt_np[:, i, :]], axis=0)
        ax.plot(seg[:, 0], seg[:, 1], "--", color="#7CFC00", lw=2.0,
                label=L["gt_short"] if i == 0 else None)
        ax.plot(obs_np[-1, i, 0], obs_np[-1, i, 1], "o", color="white", ms=7, mec="black")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_curves(scene: str, out_dir: str, fig_dir: str, L=None, show_title=True):
    """Learning curves for all three models -- useful as a report figure.

    The losses are not on the same scale (L2 vs NLL) so they are drawn on separate axes;
    the validation ADE is comparable and is drawn together.
    """
    import json

    L = L or LABELS["sr"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    ax_nll = axes[0].twinx()

    # The right panel may only contain MUTUALLY COMPARABLE curves. The LSTM baseline
    # reports deterministic ADE while the two NLL models report best-of-K ADE -- those are
    # not the same metric, so plotting them on one axis would be misleading. The right
    # panel therefore shows the controlled comparison (the two NLL models, both best-of-K),
    # while the left shows all three loss curves on separate axes (L2 and NLL are also not
    # on the same scale).
    for name, fname, color, is_nll, ade_key in [
        ("LSTM baseline (L2)", f"lstm_{scene}_log.json", C_LSTM, False, None),
        ("LSTM-prob (NLL)", f"lstm_prob_{scene}_log.json", "#8c6bb1", True, "val_ade"),
        ("Social-STGCNN (NLL)", f"stgcnn_{scene}_log.json", C_STGCNN, True, "val_ade_best"),
    ]:
        path = os.path.join(out_dir, fname)
        if not os.path.exists(path):
            continue
        hist = json.load(open(path))["history"]
        ep = [h["epoch"] for h in hist]
        target = ax_nll if is_nll else axes[0]
        target.plot(ep, [h["loss"] for h in hist], color=color, label=name,
                    ls="--" if is_nll else "-")
        if ade_key is not None:
            axes[1].plot(ep, [h[ade_key] for h in hist], color=color, label=name)

    if show_title:
        axes[0].set_title(L["loss_title"].format(scene=scene))
        axes[1].set_title(L["ade_title"].format(scene=scene))
    axes[0].set_xlabel(L["epoch"])
    axes[0].set_ylabel("L2 loss")
    ax_nll.set_ylabel("NLL")
    axes[1].set_xlabel(L["epoch"])
    axes[1].set_ylabel("best-of-5 ADE [m]")

    # combined legend covering both axes of the left subplot
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = ax_nll.get_legend_handles_labels()
    axes[0].legend(h1 + h2, l1 + l2, fontsize=8, loc="best")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.25, ls=":")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"curves_{scene}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="zara1")
    p.add_argument("--data_root", default="data")
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--fig_dir", default="outputs/figures")
    p.add_argument("--n_scenes", type=int, default=6, help="how many test sequences to render")
    p.add_argument("--min_peds", type=int, default=3, help="only scenes with at least this many pedestrians")
    p.add_argument("--max_peds", type=int, default=8, help="upper bound -- keeps the figure readable")
    p.add_argument("--min_motion", type=float, default=3.0,
                   help="min. mean path length per agent [m] -- filters out standing pedestrians")
    p.add_argument("--lang", default="sr", choices=["sr", "en"], help="language of the in-figure labels")
    p.add_argument("--no_title", action="store_true",
                   help="omit the in-figure title -- IEEE convention (description goes in the caption)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)
    L = LABELS[args.lang]
    device = args.device
    lstm, stgcnn = load_models(args.scene, args.out_dir, device)

    # batch_size=1: one scene visualised at a time
    _, loader = make_loader(os.path.join(args.data_root, args.scene), "test", False, batch_size=1)

    # We do not simply take the first N sequences -- many contain pedestrians standing
    # still, which is visually uninformative. Sequences are ranked by "interestingness":
    # number of pedestrians multiplied by mean path length, and the best-scoring are kept.
    candidates = []
    for i, batch in enumerate(loader):
        obs, gt = batch[0], batch[1]
        n_ped = obs.shape[1]
        # upper bound: scenes with 40 pedestrians are unreadable as a report figure
        if n_ped < args.min_peds or n_ped > args.max_peds:
            continue
        full = torch.cat([obs, gt], dim=0)                       # (20, V, 2)
        path_len = torch.norm(full[1:] - full[:-1], dim=-1).sum(0)  # path length per agent
        if path_len.mean().item() < args.min_motion:
            continue
        candidates.append((n_ped * path_len.mean().item(), i, batch))

    candidates.sort(key=lambda c: -c[0])
    if not candidates:
        raise SystemExit(
            f"No test sequences with >={args.min_peds} pedestrians and motion "
            f">={args.min_motion} m. Lower --min_peds or --min_motion."
        )

    made = 0
    for _, _, batch in candidates:
        obs, gt, obs_rel = batch[0], batch[1], batch[2]
        obs, gt, obs_rel = obs.to(device), gt.to(device), obs_rel.to(device)

        lstm_pred, stgcnn_pred, params = predict_both(lstm, stgcnn, obs, obs_rel, device)

        tag = f"{args.scene}_{made:02d}"
        n = obs.shape[1]
        traj_title = "" if args.no_title else L["traj_title"].format(scene=args.scene.upper(), n=n)
        heat_title = "" if args.no_title else L["heat_title"].format(scene=args.scene.upper())
        plot_trajectories(
            obs, gt, lstm_pred, stgcnn_pred,
            os.path.join(args.fig_dir, f"trajectories_{tag}.png"), traj_title, L,
        )
        plot_heatmap(
            obs, gt, params,
            os.path.join(args.fig_dir, f"heatmap_{tag}.png"), heat_title, L,
        )
        made += 1
        if made >= args.n_scenes:
            break

    plot_training_curves(args.scene, args.out_dir, args.fig_dir, L, not args.no_title)
    print(f"Saved {made} figure pairs + learning curves to {args.fig_dir}/")


if __name__ == "__main__":
    main()
