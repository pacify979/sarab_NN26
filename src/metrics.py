"""
Standardne metrike za predikciju trajektorija: ADE i FDE.

VAZNA NAPOMENA za izvestaj:
ADE NIJE srednja kvadratna greska (MSE), nego srednja EUKLIDSKA udaljenost
izmedju predvidjene i stvarne pozicije, usrednjena po svim vremenskim koracima
predikcije i po svim agentima. Zato se i izrazava u metrima -- MSE bi bio u m^2.
Ovo je definicija koja se koristi u svim referentnim radovima (Social-LSTM,
Social-GAN, Social-STGCNN), pa je bitno da se u radu navede tacno tako.
"""

import torch


def displacement_error(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """ADE po agentu.

    Args:
        pred, gt: (pred_len, N, 2) apsolutne koordinate u metrima
    Returns:
        (N,) srednja euklidska greska po agentu
    """
    return torch.norm(pred - gt, p=2, dim=-1).mean(dim=0)


def final_displacement_error(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """FDE po agentu: euklidska udaljenost u poslednjem predvidjenom koraku.

    Args:
        pred, gt: (pred_len, N, 2)
    Returns:
        (N,)
    """
    return torch.norm(pred[-1] - gt[-1], p=2, dim=-1)


def relative_to_abs(rel_traj: torch.Tensor, start_pos: torch.Tensor) -> torch.Tensor:
    """Pretvara predvidjena relativna pomeranja u apsolutne koordinate.

    Args:
        rel_traj:  (T, N, 2) pomeraji izmedju uzastopnih koraka
        start_pos: (N, 2) poslednja OSMOTRENA pozicija svakog agenta
    Returns:
        (T, N, 2) apsolutne pozicije
    """
    return torch.cumsum(rel_traj, dim=0) + start_pos.unsqueeze(0)


class ErrorAccumulator:
    """Sabira greske preko celog test skupa i racuna ispravan prosek po agentu.

    Bitno: prosek proseka po batch-evima NIJE tacan jer batch-evi imaju razlicit
    broj agenata. Zato akumuliramo sumu gresaka i broj agenata.
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
