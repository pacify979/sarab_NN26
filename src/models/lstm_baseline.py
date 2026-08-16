"""
Baseline: Vanilla LSTM (encoder-decoder), bez ikakve svesti o drugim agentima.

Svaki pesak se posmatra kao nezavisna vremenska serija -- sto je upravo poenta
baseline-a: pokazuje koliko se dobija kada se socijalne interakcije ignorisu,
i sluzi kao referenca za ST-GCNN.

Model radi nad RELATIVNIM pomeranjima (dx, dy), ne nad apsolutnim koordinatama.
Razlog: apsolutne koordinate zavise od scene (koordinatni sistem svake scene je
drugaciji), pa model nauci "gde su ljudi u ovoj sceni" umesto "kako se ljudi krecu".
Sa pomerajima je model translaciono invarijantan i mnogo bolje generalizuje na
neviđenu scenu (leave-one-scene-out protokol).
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

        # Probabilisticka varijanta predvidja parametre bivarijantne Gausove raspodele
        # (mu_x, mu_y, sigma_x, sigma_y, rho) umesto jedne tacke -- istu glavu i isti
        # NLL gubitak koristi i Social-STGCNN. Sluzi da poredjenje dva modela bude
        # posteno: bez toga bi se poredio model treniran L2 gubitkom sa modelom
        # treniranim NLL gubitkom, pa razlika ne bi merila doprinos grafa nego
        # razliku u funkciji gubitka.
        self.output_dim = 5 if probabilistic else 2

        # (dx, dy) -> vektor u embedding prostoru
        self.spatial_embedding = nn.Linear(2, embedding_dim)

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=lstm_dropout)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=lstm_dropout)
        self.hidden2pos = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, obs_traj_rel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_traj_rel: (obs_len, N, 2) osmotreni pomeraji
        Returns:
            (pred_len, N, 2) predvidjeni pomeraji, odnosno
            (pred_len, N, 5) parametri raspodele u probabilistickoj varijanti
        """
        n = obs_traj_rel.shape[1]

        # --- Encoder: sazima istoriju kretanja u skriveno stanje ---
        emb = self.spatial_embedding(obs_traj_rel)  # (obs_len, N, emb)
        _, state = self.encoder(emb)

        # --- Decoder: autoregresivno generise 12 buducih pomeraja ---
        # Krece od poslednjeg osmotrenog pomeraja i svoj izlaz vraca kao sledeci ulaz.
        last_input = obs_traj_rel[-1]  # (N, 2)
        outputs = []
        for _ in range(self.pred_len):
            dec_in = self.spatial_embedding(last_input).unsqueeze(0)  # (1, N, emb)
            out, state = self.decoder(dec_in, state)
            step = self.hidden2pos(out.squeeze(0))  # (N, output_dim)
            outputs.append(step)
            # u dekoder se vraca samo predvidjeni pomeraj (srednja vrednost),
            # ne i parametri neizvesnosti
            last_input = step[..., :2]

        return torch.stack(outputs, dim=0)  # (pred_len, N, output_dim)
