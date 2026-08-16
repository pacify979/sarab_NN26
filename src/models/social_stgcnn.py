"""
Social-STGCNN -- Spatio-Temporal Graph Convolutional Neural Network.

Adaptirano prema radu:
    Mohamed et al., "Social-STGCNN: A Social Spatio-Temporal Graph Convolutional
    Neural Network for Human Trajectory Prediction", CVPR 2020.
Referentna implementacija: github.com/abduallahmohamed/Social-STGCNN

Kljucna ideja (za izvestaj):
  1. Scena se u svakom vremenskom trenutku predstavlja kao GRAF: cvorovi su pesaci,
     a tezine grana su funkcija medjusobne udaljenosti -- blizi pesaci imaju jacu
     vezu. Time se socijalna interakcija modeluje eksplicitno, za razliku od LSTM-a.
  2. Prostorna komponenta: graph convolution nad tom matricom susedstva agregira
     informaciju o susedima u svakom vremenskom koraku.
  3. Vremenska komponenta: TCN (temporal CNN) nad vremenskom osom -- za razliku od
     LSTM-a obradjuje sve korake paralelno, pa je znatno brzi za trening.
  4. Izlaz nije jedna tacka, nego PARAMETRI bivarijantne Gausove raspodele
     (mu_x, mu_y, sigma_x, sigma_y, rho) po koraku. Model uci raspodelu verovatnoce
     kretanja -- to je ono sto omogucava "heatmap" vizuelizaciju socijalnih zona
     i sto se u evaluaciji koristi kao Best-of-N uzorkovanje.

Konvencija oblika tenzora: V = broj cvorova (pesaka), T = vreme, C = kanali.
"""

import math

import torch
import torch.nn as nn


def build_adjacency(obs_traj: torch.Tensor, seq_start_end: torch.Tensor = None) -> torch.Tensor:
    """Gradi normalizovanu matricu susedstva iz apsolutnih pozicija.

    Tezina grane = 1 / euklidska_udaljenost (bliskost). Self-loop se dodaje pre
    simetricne normalizacije A_norm = D^(-1/2) (A + I) D^(-1/2), kao u standardnom
    GCN-u (Kipf & Welling).

    Ako je prosledjen `seq_start_end`, vise scena se pakuje u JEDNU BLOK-DIJAGONALNU
    matricu: pesaci iz razlicitih scena nemaju granu medju sobom, pa je propagacija
    kroz graf identicna obradi scena jednu po jednu, ali se sve racuna u jednom GPU
    prolazu (~24x brzi trening, jer nestaje overhead lansiranja kernela za hiljade
    sitnih grafova).

    Jedina razlika u odnosu na obradu scenu-po-scenu je da BatchNorm sada racuna
    statistike preko svih cvorova u batch-u, a ne preko jedne scene. To je uobicajeno
    ponasanje mini-batch treninga (i po pravilu stabilizuje trening), ali nije
    bukvalno isti racun -- posteno je to navesti.

    Args:
        obs_traj: (T, V, 2) apsolutne koordinate
        seq_start_end: (B, 2) granice scena unutar dimenzije V; None = jedna scena
    Returns:
        (T, V, V) normalizovane matrice susedstva
    """
    T, V, _ = obs_traj.shape
    dist = torch.cdist(obs_traj, obs_traj)  # (T, V, V)

    A = torch.zeros_like(dist)
    mask = dist > 1e-6
    A[mask] = 1.0 / dist[mask]  # bliskost umesto udaljenosti

    if seq_start_end is not None:
        # nuliramo sve grane izmedju pesaka iz razlicitih scena
        block = torch.zeros(V, V, dtype=torch.bool, device=obs_traj.device)
        for s, e in seq_start_end:
            block[s:e, s:e] = True
        A = A * block

    eye = torch.eye(V, device=obs_traj.device).unsqueeze(0)
    A = A + eye  # self-loop: cvor zadrzava i sopstvenu informaciju

    deg = A.sum(dim=-1)  # (T, V)
    d_inv_sqrt = deg.pow(-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    D = torch.diag_embed(d_inv_sqrt)  # (T, V, V)
    return D @ A @ D


class GraphConv(nn.Module):
    """Prostorna graph konvolucija: 1x1 konvolucija po kanalima + agregacija preko A."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, T, V)
            A: (N, T, V, V)
        Returns:
            (N, C_out, T, V)
        """
        x = self.conv(x)  # transformacija po kanalima
        # agregacija po susedima: za svaki t, x[:, :, t, :] @ A[:, t]
        x = torch.einsum("nctv,ntvw->nctw", x, A)
        return x.contiguous()


class STGCNNLayer(nn.Module):
    """Jedan ST-GCNN blok: prostorna graph konvolucija + vremenska konvolucija (TCN)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "vremenski kernel mora biti neparan (zbog simetricnog paddinga)"
        padding = (kernel_size - 1) // 2

        self.gcn = GraphConv(in_channels, out_channels)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, (kernel_size, 1), padding=(padding, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )

        # rezidualna veza stabilizuje trening dubljih konfiguracija
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )
        self.prelu = nn.PReLU()

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.tcn(self.gcn(x, A)) + res
        return self.prelu(x)


class SocialSTGCNN(nn.Module):
    """Kompletan model: n_stgcnn ST-GCNN blokova + n_txpcnn slojeva za ekstrapolaciju vremena."""

    def __init__(
        self,
        n_stgcnn: int = 1,
        n_txpcnn: int = 5,
        input_feat: int = 2,
        output_feat: int = 5,   # mu_x, mu_y, sigma_x, sigma_y, rho
        seq_len: int = 8,
        pred_seq_len: int = 12,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_stgcnn = n_stgcnn
        self.n_txpcnn = n_txpcnn
        self.pred_seq_len = pred_seq_len

        # --- prostorno-vremenski enkoder ---
        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(STGCNNLayer(input_feat, output_feat, kernel_size, dropout))
        for _ in range(1, n_stgcnn):
            self.st_gcns.append(STGCNNLayer(output_feat, output_feat, kernel_size, dropout))

        # --- TXP-CNN: prosiruje vremensku osu sa 8 osmotrenih na 12 predvidjenih koraka ---
        # Radi konvolucijom PO VREMENSKOJ dimenziji (a ne rekurentno kao LSTM dekoder),
        # pa se svih 12 koraka generise u jednom prolazu -- otud brzina modela.
        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(seq_len, pred_seq_len, 3, padding=1))
        for _ in range(1, n_txpcnn):
            self.tpcnns.append(nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1))
        self.tpcnn_output = nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1)
        self.prelus = nn.ModuleList([nn.PReLU() for _ in range(n_txpcnn)])

    def forward(self, v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Args:
            v: (N, C_in, T_obs, V) ulazne osobine cvorova (relativna pomeranja)
            a: (N, T_obs, V, V) matrice susedstva
        Returns:
            (N, 5, T_pred, V) parametri bivarijantne Gausove raspodele
        """
        for layer in self.st_gcns:
            v = layer(v, a)

        # zamena osa: vreme postaje kanalska dimenzija da bismo konvoluirali po vremenu
        v = v.permute(0, 2, 1, 3)  # (N, T, C, V)

        v = self.prelus[0](self.tpcnns[0](v))
        for i in range(1, self.n_txpcnn):
            v = self.prelus[i](self.tpcnns[i](v)) + v  # rezidualne veze
        v = self.tpcnn_output(v)

        return v.permute(0, 2, 1, 3)  # nazad u (N, C, T_pred, V)


def bivariate_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood bivarijantne Gausove raspodele.

    Model ne uci samo gde ce pesak biti, nego i KOLIKO JE SIGURAN u to.
    Sigma raste u situacijama sa vise mogucih ishoda (raskrsnice, izbegavanje sudara)
    -- upravo to koristimo za heatmap vizuelizaciju socijalnih zona.

    Args:
        pred:   (T, V, 5) -> mu_x, mu_y, log_sigma_x, log_sigma_y, rho_raw
        target: (T, V, 2) stvarna relativna pomeranja
    """
    norm_x = target[:, :, 0] - pred[:, :, 0]
    norm_y = target[:, :, 1] - pred[:, :, 1]

    # exp / tanh garantuju validne parametre: sigma > 0, |rho| < 1
    sx = torch.exp(pred[:, :, 2]).clamp(min=1e-3)
    sy = torch.exp(pred[:, :, 3]).clamp(min=1e-3)
    corr = torch.tanh(pred[:, :, 4]).clamp(min=-0.99, max=0.99)

    sxsy = sx * sy
    z = (norm_x / sx) ** 2 + (norm_y / sy) ** 2 - 2 * corr * norm_x * norm_y / sxsy
    neg_rho = 1 - corr**2

    result = torch.exp(-z / (2 * neg_rho))
    denom = 2 * math.pi * sxsy * torch.sqrt(neg_rho)
    result = -torch.log(torch.clamp(result / denom, min=1e-9))
    return result.mean()
