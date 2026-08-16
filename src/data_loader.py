"""
Data loader za ETH-UCY skup podataka.

Format ulaznih fajlova (tab-separated):
    frame_id    ped_id    x    y

Iz svake scene se izvlace sekvence duzine obs_len + pred_len (8 + 12 = 20 frejmova).
Zadrzavaju se samo pesaci koji su prisutni u SVIH 20 frejmova sekvence -- to je
standardni protokol iz Social-GAN / Social-STGCNN radova, pa su rezultati uporedivi
sa brojkama iz literature.

Isti loader koristi i LSTM baseline i ST-GCNN model: LSTM ignorise `seq_start_end`,
dok ST-GCNN preko njega zna koji pesaci pripadaju istoj sceni (tj. istom grafu).
"""

import os
import pickle
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def read_file(path: str) -> np.ndarray:
    """Ucitava jedan .txt fajl u niz oblika (N, 4) = [frame, ped_id, x, y]."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            rows.append([float(p) for p in parts[:4]])
    return np.asarray(rows, dtype=np.float32)


class TrajectoryDataset(Dataset):
    """Jedna instanca = jedna vremenska sekvenca sa svim pesacima u njoj.

    Vraca tuple tenzora oblika (n_ped, 2, T):
        obs_traj       apsolutne koordinate, prvih obs_len koraka
        pred_traj      apsolutne koordinate, narednih pred_len koraka (ground truth)
        obs_traj_rel   relativna pomeranja (razlike izmedju uzastopnih tacaka)
        pred_traj_rel  relativna pomeranja za buducnost
    """

    def __init__(
        self,
        data_dir: str,
        obs_len: int = 8,
        pred_len: int = 12,
        skip: int = 1,
        min_ped: int = 1,
        cache_dir: str = ".cache",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len

        # Parsiranje je sporo (univ ima ~100k redova), pa kesiramo rezultat na disk.
        # Optuna pokrece desetine trial-ova -- bez kesa bi svaki ponovo parsirao podatke.
        cache_key = f"{data_dir.strip(os.sep).replace(os.sep, '_')}_{obs_len}_{pred_len}_{skip}_{min_ped}.pkl"
        cache_path = os.path.join(cache_dir, cache_key)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                seq_list, seq_list_rel, self.seq_start_end = pickle.load(f)
        else:
            seq_list, seq_list_rel, self.seq_start_end = self._build()
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump((seq_list, seq_list_rel, self.seq_start_end), f)

        # Sve trajektorije iz svih sekvenci su spojene u jedan veliki tenzor;
        # seq_start_end cuva granice pojedinacnih sekvenci u tom tenzoru.
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)

        self.obs_traj = torch.from_numpy(seq_list[:, :, : self.obs_len]).float()
        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len :]).float()
        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, : self.obs_len]).float()
        self.pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len :]).float()
        self.num_peds = self.obs_traj.shape[0]

    def _build(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[int, int]]]:
        files = sorted(
            os.path.join(self.data_dir, f)
            for f in os.listdir(self.data_dir)
            if f.endswith(".txt")
        )
        if not files:
            raise FileNotFoundError(f"Nema .txt fajlova u {self.data_dir}")

        seq_list, seq_list_rel, num_peds_in_seq = [], [], []

        for path in files:
            data = read_file(path)
            frames = np.unique(data[:, 0]).tolist()
            # mapa frame -> indeks, da izbegnemo O(n) pretragu u unutrasnjoj petlji
            frame_to_idx = {f: i for i, f in enumerate(frames)}
            frame_data = [data[data[:, 0] == f, :] for f in frames]

            for idx in range(0, len(frames) - self.seq_len + 1, 1):
                curr_seq_data = np.concatenate(frame_data[idx : idx + self.seq_len], axis=0)
                peds_in_seq = np.unique(curr_seq_data[:, 1])

                curr_seq = np.zeros((len(peds_in_seq), 2, self.seq_len), dtype=np.float32)
                curr_seq_rel = np.zeros_like(curr_seq)
                n_considered = 0

                for ped_id in peds_in_seq:
                    ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    pad_front = frame_to_idx[ped_seq[0, 0]] - idx
                    pad_end = frame_to_idx[ped_seq[-1, 0]] - idx + 1
                    # pesak mora da postoji u svih seq_len uzastopnih frejmova
                    if pad_end - pad_front != self.seq_len or ped_seq.shape[0] != self.seq_len:
                        continue

                    xy = ped_seq[:, 2:4].T  # (2, seq_len)
                    rel = np.zeros_like(xy)
                    rel[:, 1:] = xy[:, 1:] - xy[:, :-1]

                    curr_seq[n_considered] = xy
                    curr_seq_rel[n_considered] = rel
                    n_considered += 1

                if n_considered >= 1:
                    num_peds_in_seq.append(n_considered)
                    seq_list.append(curr_seq[:n_considered])
                    seq_list_rel.append(curr_seq_rel[:n_considered])

        cum = np.cumsum([0] + num_peds_in_seq).tolist()
        seq_start_end = [(s, e) for s, e in zip(cum[:-1], cum[1:])]
        return seq_list, seq_list_rel, seq_start_end

    def __len__(self) -> int:
        return len(self.seq_start_end)

    def __getitem__(self, index: int):
        start, end = self.seq_start_end[index]
        return (
            self.obs_traj[start:end],
            self.pred_traj[start:end],
            self.obs_traj_rel[start:end],
            self.pred_traj_rel[start:end],
        )


def seq_collate(batch):
    """Spaja vise sekvenci u batch.

    Broj pesaka po sceni je promenljiv, pa ne mozemo u klasican (B, ...) tenzor.
    Umesto toga sve pesake nizemo po dimenziji N i pamtimo granice scena u
    `seq_start_end` -- konvencija iz Social-GAN-a.

    Izlaz:
        obs_traj      (obs_len, N, 2)
        pred_traj     (pred_len, N, 2)
        obs_traj_rel  (obs_len, N, 2)
        pred_traj_rel (pred_len, N, 2)
        seq_start_end (B, 2)
    """
    obs_list, pred_list, obs_rel_list, pred_rel_list = zip(*batch)

    counts = [seq.shape[0] for seq in obs_list]
    cum = np.cumsum([0] + counts).tolist()
    seq_start_end = torch.tensor(
        [[s, e] for s, e in zip(cum[:-1], cum[1:])], dtype=torch.long
    )

    def cat(seqs):
        # (n_ped, 2, T) -> (T, N, 2)
        return torch.cat(seqs, dim=0).permute(2, 0, 1)

    return cat(obs_list), cat(pred_list), cat(obs_rel_list), cat(pred_rel_list), seq_start_end


def data_loader(
    data_dir: str,
    split: str,
    obs_len: int = 8,
    pred_len: int = 12,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
):
    """Pomocna funkcija: npr. data_loader('data/eth', 'train')."""
    from torch.utils.data import DataLoader

    dset = TrajectoryDataset(
        os.path.join(data_dir, split), obs_len=obs_len, pred_len=pred_len
    )
    loader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=seq_collate,
    )
    return dset, loader
