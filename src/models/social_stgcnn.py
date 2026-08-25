"""
Social-STGCNN -- Spatio-Temporal Graph Convolutional Neural Network.

Re-implemented from the paper:
    Mohamed et al., "Social-STGCNN: A Social Spatio-Temporal Graph Convolutional
    Neural Network for Human Trajectory Prediction", CVPR 2020.
Reference implementation: github.com/abduallahmohamed/Social-STGCNN

Key ideas (for the report):
  1. At every timestep the scene is represented as a GRAPH: nodes are pedestrians and
     edge weights are a function of mutual distance -- closer pedestrians are more
     strongly connected. Social interaction is therefore modelled explicitly, unlike
     in the LSTM.
  2. Spatial component: a graph convolution over that adjacency matrix aggregates
     information about neighbours at each timestep.
  3. Temporal component: a TCN (temporal CNN) along the time axis -- unlike an LSTM it
     processes all steps in parallel, which makes training considerably faster.
  4. The output is not a single point but the PARAMETERS of a bivariate Gaussian
     (mu_x, mu_y, sigma_x, sigma_y, rho) per step. The model learns a probability
     distribution over motion -- this is what enables the "heatmap" visualisation of
     social zones and what the Best-of-N sampling protocol evaluates.

NOTE on a deliberate deviation from the reference implementation: it normalises the
adjacency matrix with the normalised graph Laplacian (L = I - D^-1/2 A D^-1/2), whereas
we use the normalised adjacency D^-1/2 (A + I) D^-1/2, i.e. the standard GCN propagation
rule of Kipf & Welling. Since L = I - A_hat these are different operators.

Tensor shape convention: V = number of nodes (pedestrians), T = time, C = channels.
"""

import math

import torch
import torch.nn as nn


def build_adjacency(obs_traj: torch.Tensor, seq_start_end: torch.Tensor = None) -> torch.Tensor:
    """Builds the normalised adjacency matrix from absolute positions.

    Edge weight = 1 / euclidean_distance (proximity). A self-loop is added before the
    symmetric normalisation A_norm = D^(-1/2) (A + I) D^(-1/2), as in the standard GCN
    formulation (Kipf & Welling).

    If `seq_start_end` is given, several scenes are packed into ONE BLOCK-DIAGONAL
    matrix: pedestrians from different scenes have no edge between them, so graph
    propagation is identical to processing the scenes one at a time, while everything is
    computed in a single GPU pass (~24x faster training, because the kernel-launch
    overhead of thousands of tiny graphs disappears).

    The one real difference from scene-by-scene processing is that BatchNorm now computes
    its statistics over all nodes in the batch rather than over a single scene. That is
    ordinary mini-batch behaviour (and generally stabilises training), but it is not a
    bit-identical computation -- worth stating honestly.

    Args:
        obs_traj: (T, V, 2) absolute coordinates
        seq_start_end: (B, 2) scene boundaries along dimension V; None = a single scene
    Returns:
        (T, V, V) normalised adjacency matrices
    """
    T, V, _ = obs_traj.shape
    dist = torch.cdist(obs_traj, obs_traj)  # (T, V, V)

    A = torch.zeros_like(dist)
    mask = dist > 1e-6
    A[mask] = 1.0 / dist[mask]  # proximity instead of distance

    if seq_start_end is not None:
        # zero out every edge between pedestrians from different scenes
        block = torch.zeros(V, V, dtype=torch.bool, device=obs_traj.device)
        for s, e in seq_start_end:
            block[s:e, s:e] = True
        A = A * block

    eye = torch.eye(V, device=obs_traj.device).unsqueeze(0)
    A = A + eye  # self-loop: a node keeps its own information as well

    deg = A.sum(dim=-1)  # (T, V)
    d_inv_sqrt = deg.pow(-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    D = torch.diag_embed(d_inv_sqrt)  # (T, V, V)
    return D @ A @ D


class GraphConv(nn.Module):
    """Spatial graph convolution: a 1x1 channel-wise convolution + aggregation over A."""

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
        x = self.conv(x)  # channel-wise transformation
        # neighbour aggregation: for every t, x[:, :, t, :] @ A[:, t]
        x = torch.einsum("nctv,ntvw->nctw", x, A)
        return x.contiguous()


class STGCNNLayer(nn.Module):
    """One ST-GCNN block: spatial graph convolution + temporal convolution (TCN)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "temporal kernel must be odd (for symmetric padding)"
        padding = (kernel_size - 1) // 2

        self.gcn = GraphConv(in_channels, out_channels)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, (kernel_size, 1), padding=(padding, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )

        # the residual connection stabilises training of deeper configurations
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
    """Full model: n_stgcnn ST-GCNN blocks + n_txpcnn layers for temporal extrapolation."""

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

        # --- spatio-temporal encoder ---
        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(STGCNNLayer(input_feat, output_feat, kernel_size, dropout))
        for _ in range(1, n_stgcnn):
            self.st_gcns.append(STGCNNLayer(output_feat, output_feat, kernel_size, dropout))

        # --- TXP-CNN: expands the time axis from 8 observed to 12 predicted steps ---
        # It works by convolving ALONG THE TIME dimension (rather than recurrently, as an
        # LSTM decoder would), so all 12 steps are produced in one pass -- hence the speed.
        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(seq_len, pred_seq_len, 3, padding=1))
        for _ in range(1, n_txpcnn):
            self.tpcnns.append(nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1))
        self.tpcnn_output = nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1)
        self.prelus = nn.ModuleList([nn.PReLU() for _ in range(n_txpcnn)])

    def forward(self, v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Args:
            v: (N, C_in, T_obs, V) input node features (relative displacements)
            a: (N, T_obs, V, V) adjacency matrices
        Returns:
            (N, 5, T_pred, V) bivariate Gaussian parameters
        """
        for layer in self.st_gcns:
            v = layer(v, a)

        # swap axes: time becomes the channel dimension so we can convolve over time
        v = v.permute(0, 2, 1, 3)  # (N, T, C, V)

        v = self.prelus[0](self.tpcnns[0](v))
        for i in range(1, self.n_txpcnn):
            v = self.prelus[i](self.tpcnns[i](v)) + v  # residual connections
        v = self.tpcnn_output(v)

        return v.permute(0, 2, 1, 3)  # back to (N, C, T_pred, V)


def bivariate_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of a bivariate Gaussian.

    The model learns not only where a pedestrian will be, but also HOW CONFIDENT it is.
    Sigma grows in situations with several plausible outcomes (junctions, collision
    avoidance) -- which is exactly what the heatmap visualisation of social zones shows.

    Args:
        pred:   (T, V, 5) -> mu_x, mu_y, log_sigma_x, log_sigma_y, rho_raw
        target: (T, V, 2) true relative displacements
    """
    norm_x = target[:, :, 0] - pred[:, :, 0]
    norm_y = target[:, :, 1] - pred[:, :, 1]

    # exp / tanh guarantee valid parameters: sigma > 0, |rho| < 1
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
