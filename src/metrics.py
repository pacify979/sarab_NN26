"""
Standard trajectory-prediction metrics: ADE and FDE.

IMPORTANT NOTE for the report:
ADE is NOT mean squared error. It is the mean EUCLIDEAN distance between the
predicted and the true position, averaged over all predicted timesteps and all
agents. That is why it is expressed in metres -- MSE would be in m^2. This is the
definition used in every reference paper (Social-LSTM, Social-GAN, Social-STGCNN),
so the report must state it exactly this way.
"""

import torch


def displacement_error(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Per-agent ADE.

    Args:
        pred, gt: (pred_len, N, 2) absolute coordinates in metres
    Returns:
        (N,) mean Euclidean error per agent
    """
    return torch.norm(pred - gt, p=2, dim=-1).mean(dim=0)


def final_displacement_error(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Per-agent FDE: Euclidean distance at the final predicted step.

    Args:
        pred, gt: (pred_len, N, 2)
    Returns:
        (N,)
    """
    return torch.norm(pred[-1] - gt[-1], p=2, dim=-1)


def relative_to_abs(rel_traj: torch.Tensor, start_pos: torch.Tensor) -> torch.Tensor:
    """Converts predicted relative displacements back to absolute coordinates.

    Args:
        rel_traj:  (T, N, 2) displacements between consecutive steps
        start_pos: (N, 2) last OBSERVED position of each agent
    Returns:
        (T, N, 2) absolute positions
    """
    return torch.cumsum(rel_traj, dim=0) + start_pos.unsqueeze(0)


class ErrorAccumulator:
    """Accumulates errors over the whole test set and computes a correct per-agent mean.

    Important: averaging per-batch means is NOT correct, because batches contain
    different numbers of agents. We therefore accumulate the error sum and the
    agent count, and divide only at the end.
    """

    def __init__(self):
        self.ade_sum = 0.0
        self.fde_sum = 0.0
        self.n = 0

    def update(self, pred: torch.Tensor, gt: torch.Tensor) -> None:
        self.ade_sum += displacement_error(pred, gt).sum().item()
        self.fde_sum += final_displacement_error(pred, gt).sum().item()
        self.n += pred.shape[1]

    @property
    def ade(self) -> float:
        return self.ade_sum / max(self.n, 1)

    @property
    def fde(self) -> float:
        return self.fde_sum / max(self.n, 1)
