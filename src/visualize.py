"""
Kvalitativna analiza: vizuelizacija predvidjenih trajektorija i heatmap-a verovatnoce.

Primer:
    python -m src.visualize --scene zara1 --n_scenes 6
    python -m src.visualize --scene eth --n_scenes 4 --min_peds 4

Generise dva tipa slika u outputs/figures/:
  1. trajectories_<scena>_<i>.png -- osmotreni deo, ground truth, LSTM i ST-GCNN predikcija
  2. heatmap_<scena>_<i>.png      -- gustina verovatnoce buduceg kretanja (ST-GCNN),
     dobijena uzorkovanjem iz naucene bivarijantne raspodele. Ovo je vizuelni dokaz
     da model uci NEIZVESNOST: gustina je uska kada je kretanje predvidivo, a siri se
     i "obilazi" druge pesake u situacijama gde se postuje licni prostor.
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

# paleta: dovoljno kontrastna i za stampu u sivim tonovima
C_OBS = "#2b2b2b"
C_GT = "#1a9850"
C_LSTM = "#e08214"
C_STGCNN = "#3b6fd4"


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
    """Vraca (lstm_abs, stgcnn_abs, stgcnn_params) za jednu scenu."""
    lstm_abs = relative_to_abs(lstm(obs_rel), obs[-1])

    v, a = prepare_inputs(obs, obs_rel)
    params = stgcnn(v, a).squeeze(0).permute(1, 2, 0)  # (T_pred, V, 5)
    stgcnn_abs = relative_to_abs(params[..., 0:2], obs[-1])
    return lstm_abs, stgcnn_abs, params


def plot_trajectories(obs, gt, lstm_pred, stgcnn_pred, path: str, title: str):
    """Poredjenje predikcija svih agenata u jednoj sceni."""
    obs, gt = obs.cpu().numpy(), gt.cpu().numpy()
    lstm_pred, stgcnn_pred = lstm_pred.cpu().numpy(), stgcnn_pred.cpu().numpy()
    n_ped = obs.shape[1]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for i in range(n_ped):
        # istorija: puna linija sa markerom na startu
        ax.plot(obs[:, i, 0], obs[:, i, 1], "-", color=C_OBS, lw=2, alpha=0.85,
                label="Osmotreno (8 frejmova)" if i == 0 else None)
        ax.plot(obs[0, i, 0], obs[0, i, 1], "o", color=C_OBS, ms=5)

        # spajamo poslednju osmotrenu tacku sa svakom predikcijom da linije ne "vise"
        def seg(pred):
            return np.concatenate([obs[-1:, i, :], pred[:, i, :]], axis=0)

        ax.plot(*seg(gt).T, "--", color=C_GT, lw=2.2,
                label="Ground truth (12 frejmova)" if i == 0 else None)
        ax.plot(*seg(lstm_pred).T, "-", color=C_LSTM, lw=1.8, alpha=0.9,
                label="LSTM baseline" if i == 0 else None)
        ax.plot(*seg(stgcnn_pred).T, "-", color=C_STGCNN, lw=1.8, alpha=0.9,
                label="Social-STGCNN" if i == 0 else None)

        ax.plot(gt[-1, i, 0], gt[-1, i, 1], "*", color=C_GT, ms=12)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25, ls=":")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_heatmap(obs, gt, params, path: str, title: str, n_samples: int = 2000, bins: int = 140):
    """Heatmap gustine buduceg kretanja, dobijen uzorkovanjem iz naucene raspodele.

    Uzorkujemo veliki broj trajektorija iz bivarijantne Gausove raspodele koju je
    model predvideo, pa pravimo 2D histogram njihovih pozicija. Rezultat je mapa
    "gde ce se pesaci verovatno naci u naredne 4.8 sekunde" -- upravo ono sto
    planeru putanje robota treba da bi izbegao socijalno neprihvatljive zone.
    """
    samples = sample_trajectories(params, n_samples)          # (K, T, V, 2)
    abs_samples = torch.cumsum(samples, dim=1) + obs[-1]      # (K, T, V, 2)
    pts = abs_samples.reshape(-1, 2).cpu().numpy()

    obs_np, gt_np = obs.cpu().numpy(), gt.cpu().numpy()

    # granice prikaza obuhvataju i istoriju i uzorke
    all_x = np.concatenate([pts[:, 0], obs_np[..., 0].ravel(), gt_np[..., 0].ravel()])
    all_y = np.concatenate([pts[:, 1], obs_np[..., 1].ravel(), gt_np[..., 1].ravel()])
    pad = 1.0
    rng = [[all_x.min() - pad, all_x.max() + pad], [all_y.min() - pad, all_y.max() + pad]]

    H, xe, ye = np.histogram2d(pts[:, 0], pts[:, 1], bins=bins, range=rng)
    H = gaussian_filter(H, sigma=2.0)  # blago zaglađivanje da histogram ne bude zrnast

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(
        H.T, origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]],
        cmap="magma", aspect="auto", alpha=0.92,
    )
    fig.colorbar(im, ax=ax, label="relativna gustina verovatnoce")

    n_ped = obs_np.shape[1]
    for i in range(n_ped):
        ax.plot(obs_np[:, i, 0], obs_np[:, i, 1], "-", color="white", lw=2.2,
                label="Osmotreno" if i == 0 else None)
        seg = np.concatenate([obs_np[-1:, i, :], gt_np[:, i, :]], axis=0)
        ax.plot(seg[:, 0], seg[:, 1], "--", color="#7CFC00", lw=2.0,
                label="Ground truth" if i == 0 else None)
        ax.plot(obs_np[-1, i, 0], obs_np[-1, i, 1], "o", color="white", ms=7, mec="black")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_curves(scene: str, out_dir: str, fig_dir: str):
    """Krive ucenja za oba modela -- korisno kao slika u izvestaju."""
    import json

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, fname, color in [
        ("LSTM baseline", f"lstm_{scene}_log.json", C_LSTM),
        ("Social-STGCNN", f"stgcnn_{scene}_log.json", C_STGCNN),
    ]:
        path = os.path.join(out_dir, fname)
        if not os.path.exists(path):
            continue
        hist = json.load(open(path))["history"]
        ep = [h["epoch"] for h in hist]
        axes[0].plot(ep, [h["loss"] for h in hist], color=color, label=name)
        key = "val_ade" if "val_ade" in hist[0] else "val_ade_det"
        axes[1].plot(ep, [h[key] for h in hist], color=color, label=name)

    axes[0].set_title(f"Funkcija gubitka ({scene})")
    axes[0].set_xlabel("epoha")
    axes[0].set_ylabel("loss")
    axes[1].set_title(f"Validacioni ADE ({scene})")
    axes[1].set_xlabel("epoha")
    axes[1].set_ylabel("ADE [m]")
    for ax in axes:
        ax.grid(alpha=0.25, ls=":")
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"curves_{scene}.png"), dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="zara1")
    p.add_argument("--data_root", default="data")
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--fig_dir", default="outputs/figures")
    p.add_argument("--n_scenes", type=int, default=6, help="koliko test sekvenci vizuelizovati")
    p.add_argument("--min_peds", type=int, default=3, help="samo scene sa bar toliko pesaka")
    p.add_argument("--max_peds", type=int, default=8, help="gornji limit -- citljivost slike")
    p.add_argument("--min_motion", type=float, default=3.0,
                   help="min. prosecna duzina putanje po agentu [m] -- filtrira pesake koji stoje")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)
    device = args.device
    lstm, stgcnn = load_models(args.scene, args.out_dir, device)

    # batch_size=1: vizuelizujemo scenu po scenu
    _, loader = make_loader(os.path.join(args.data_root, args.scene), "test", False, batch_size=1)

    # Ne uzimamo prvih N sekvenci -- mnoge sadrze pesake koji stoje u mestu, sto je
    # vizuelno neinformativno. Sekvence rangiramo po "zanimljivosti": broj pesaka
    # pomnozen prosecnom predjenom putanjom, pa uzimamo najbolje ocenjene.
    candidates = []
    for i, batch in enumerate(loader):
        obs, gt = batch[0], batch[1]
        n_ped = obs.shape[1]
        # gornji limit: scene sa 40 pesaka su nečitljive na slici u izvestaju
        if n_ped < args.min_peds or n_ped > args.max_peds:
            continue
        full = torch.cat([obs, gt], dim=0)                       # (20, V, 2)
        path_len = torch.norm(full[1:] - full[:-1], dim=-1).sum(0)  # duzina putanje po agentu
        if path_len.mean().item() < args.min_motion:
            continue
        candidates.append((n_ped * path_len.mean().item(), i, batch))

    candidates.sort(key=lambda c: -c[0])
    if not candidates:
        raise SystemExit(
            f"Nema test sekvenci sa >={args.min_peds} pesaka i kretanjem >={args.min_motion} m. "
            "Smanji --min_peds ili --min_motion."
        )

    made = 0
    for _, _, batch in candidates:
        obs, gt, obs_rel = batch[0], batch[1], batch[2]
        obs, gt, obs_rel = obs.to(device), gt.to(device), obs_rel.to(device)

        lstm_pred, stgcnn_pred, params = predict_both(lstm, stgcnn, obs, obs_rel, device)

        tag = f"{args.scene}_{made:02d}"
        n = obs.shape[1]
        plot_trajectories(
            obs, gt, lstm_pred, stgcnn_pred,
            os.path.join(args.fig_dir, f"trajectories_{tag}.png"),
            f"Scena {args.scene.upper()} -- {n} pesaka: predikcija vs. ground truth",
        )
        plot_heatmap(
            obs, gt, params,
            os.path.join(args.fig_dir, f"heatmap_{tag}.png"),
            f"Scena {args.scene.upper()} -- gustina verovatnoce kretanja (Social-STGCNN)",
        )
        made += 1
        if made >= args.n_scenes:
            break

    plot_training_curves(args.scene, args.out_dir, args.fig_dir)
    print(f"Sacuvano {made} para slika + krive ucenja u {args.fig_dir}/")


if __name__ == "__main__":
    main()
