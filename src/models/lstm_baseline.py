"""
Baseline: Vanilla LSTM (encoder-decoder), with no awareness of other agents.

Each pedestrian is treated as an independent time series -- which is exactly the point
of the baseline: it shows how much is gained when social interactions are ignored, and
serves as the reference against which ST-GCNN is measured.

The model operates on RELATIVE displacements (dx, dy), not on absolute coordinates.
Reason: absolute coordinates are scene-dependent (every scene has its own coordinate
frame), so the model would learn "where people stand in this scene" instead of "how
people move". With displacements the model is translation invariant and generalises far
better to an unseen scene (the leave-one-scene-out protocol).
"""

import torch
import torch.nn as nn


class VanillaLSTM(nn.Module):
    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 12,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        probabilistic: bool = False,
    ):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.probabilistic = probabilistic

        # The probabilistic variant predicts the parameters of a bivariate Gaussian
        # (mu_x, mu_y, sigma_x, sigma_y, rho) instead of a single point -- the same head
        # and the same NLL loss that Social-STGCNN uses. It exists so that the comparison
        # between the two models is fair: without it we would be comparing a model trained
        # with an L2 loss against one trained with NLL, so the difference would measure the
        # change of loss function rather than the contribution of the graph.
        self.output_dim = 5 if probabilistic else 2

        # (dx, dy) -> vector in embedding space
        self.spatial_embedding = nn.Linear(2, embedding_dim)

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=lstm_dropout)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=lstm_dropout)
        self.hidden2pos = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, obs_traj_rel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_traj_rel: (obs_len, N, 2) observed displacements
        Returns:
            (pred_len, N, 2) predicted displacements, or
            (pred_len, N, 5) distribution parameters in the probabilistic variant
        """
        n = obs_traj_rel.shape[1]

        # --- Encoder: compresses the motion history into a hidden state ---
        emb = self.spatial_embedding(obs_traj_rel)  # (obs_len, N, emb)
        _, state = self.encoder(emb)

        # --- Decoder: autoregressively generates 12 future displacements ---
        # It starts from the last observed displacement and feeds its own output back
        # in as the next input.
        last_input = obs_traj_rel[-1]  # (N, 2)
        outputs = []
        for _ in range(self.pred_len):
            dec_in = self.spatial_embedding(last_input).unsqueeze(0)  # (1, N, emb)
            out, state = self.decoder(dec_in, state)
            step = self.hidden2pos(out.squeeze(0))  # (N, output_dim)
            outputs.append(step)
            # only the predicted displacement (the mean) is fed back into the decoder,
            # not the uncertainty parameters
            last_input = step[..., :2]

        return torch.stack(outputs, dim=0)  # (pred_len, N, output_dim)
