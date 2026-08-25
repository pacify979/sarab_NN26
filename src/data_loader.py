"""
Data loader for the ETH-UCY dataset.

Input file format (tab-separated):
    frame_id    ped_id    x    y

Sequences of length obs_len + pred_len (8 + 12 = 20 frames) are extracted from each
scene. Only pedestrians present in ALL 20 frames of a sequence are retained -- this is
the standard protocol from the Social-GAN / Social-STGCNN papers, which keeps our
results comparable to the figures reported in the literature.

The same loader serves both the LSTM baseline and the ST-GCNN model: the LSTM ignores
`seq_start_end` entirely, while the ST-GCNN uses it to know which pedestrians belong to
the same scene (i.e. the same graph).
"""

import os
import pickle
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def read_file(path: str) -> np.ndarray:
    """Loads one .txt file into an array of shape (N, 4) = [frame, ped_id, x, y]."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            rows.append([float(p) for p in parts[:4]])
    return np.asarray(rows, dtype=np.float32)


class TrajectoryDataset(Dataset):
    """One item = one time window containing every pedestrian present in it.

    Returns a tuple of tensors of shape (n_ped, 2, T):
        obs_traj       absolute coordinates, first obs_len steps
        pred_traj      absolute coordinates, next pred_len steps (ground truth)
        obs_traj_rel   relative displacements (differences between consecutive points)
        pred_traj_rel  relative displacements for the future
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

        # Parsing is slow (univ has ~100k rows), so the result is cached on disk.
        # Optuna runs dozens of trials -- without the cache each would re-parse the data.
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

        # All trajectories from all sequences are concatenated into one large tensor;
        # seq_start_end stores the boundaries of the individual sequences within it.
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
            raise FileNotFoundError(f"No .txt files found in {self.data_dir}")

        seq_list, seq_list_rel, num_peds_in_seq = [], [], []

        for path in files:
            data = read_file(path)
            frames = np.unique(data[:, 0]).tolist()
            # frame -> index map, to avoid an O(n) list lookup in the inner loop
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
                    # the pedestrian must be present in all seq_len consecutive frames
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
    """Collates several sequences into a batch.

    The number of pedestrians per scene varies, so a standard (B, ...) tensor does
    not fit. Instead every pedestrian is concatenated along dimension N and the
    scene boundaries are recorded in `seq_start_end` -- the Social-GAN convention.

    Returns:
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
    """Convenience helper, e.g. data_loader('data/eth', 'train')."""
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
