# Results

Protocol: 8 observed frames (3.2 s) -> 12 predicted frames (4.8 s), leave-one-scene-out. All values in metres; lower is better.

Training budget: LSTM baseline 100 epochs, LSTM-prob 250 epochs, Social-STGCNN 250 epochs. Checkpoints are selected by validation ADE in every case.

## Comparison 1: deterministic models (single predicted path)

| Scene | LSTM ADE | LSTM FDE | ST-GCNN ADE | ST-GCNN FDE |
|---|---|---|---|---|
| ETH | 1.007 | 1.996 | 1.043 | 2.113 |
| HOTEL | 0.457 | 0.936 | 0.436 | 0.880 |
| UNIV | 0.577 | 1.278 | 0.568 | 1.202 |
| ZARA1 | 0.402 | 0.886 | 0.419 | 0.904 |
| ZARA2 | 0.316 | 0.699 | 0.350 | 0.761 |
| **AVERAGE** | **0.552** | **1.159** | **0.563** | **1.172** |

> Note: this comparison is only conditionally fair. Social-STGCNN is trained with an NLL loss (it learns a distribution), so its mean is not optimised for ADE. The -2.1% difference therefore also measures the change of loss function, not the contribution of the graph alone.


## Comparison 2: probabilistic models (identical NLL loss, best-of-20)

This is the **key comparison**: both models use an identical bivariate Gaussian head, the same loss and the same evaluation protocol. The only difference is whether the model can see other agents, so the difference measures the contribution of social modelling alone.

Both models were trained for the same number of epochs (LSTM-prob 250, Social-STGCNN 250), so the training budget is not a confounding variable either.

| Scene | LSTM-prob ADE | LSTM-prob FDE | ST-GCNN ADE | ST-GCNN FDE |
|---|---|---|---|---|
| ETH | 0.775 | 1.405 | 0.819 | 1.582 |
| HOTEL | 0.342 | 0.659 | 0.311 | 0.561 |
| UNIV | 0.411 | 0.816 | 0.412 | 0.803 |
| ZARA1 | 0.276 | 0.497 | 0.283 | 0.498 |
| ZARA2 | 0.221 | 0.428 | 0.251 | 0.474 |
| **AVERAGE** | **0.405** | **0.761** | **0.415** | **0.784** |

**Difference, ST-GCNN vs. LSTM-prob:** ADE -2.5% (positive = ST-GCNN better)


## Published reference values

Our results should land in the same range -- this confirms that the protocol and the evaluation are implemented correctly.


**LSTM baseline (Social-GAN paper, deterministic)**

| Scene | ETH | HOTEL | UNIV | ZARA1 | ZARA2 | Average |
|---|---|---|---|---|---|---|
| ADE | 1.09 | 0.86 | 0.61 | 0.41 | 0.52 | 0.70 |
| FDE | 2.94 | 1.91 | 1.31 | 0.88 | 1.11 | 1.63 |

**Social-STGCNN (CVPR 2020, best-of-20)**

| Scene | ETH | HOTEL | UNIV | ZARA1 | ZARA2 | Average |
|---|---|---|---|---|---|---|
| ADE | 0.64 | 0.49 | 0.44 | 0.34 | 0.30 | 0.44 |
| FDE | 1.11 | 0.85 | 0.79 | 0.53 | 0.48 | 0.75 |

## Methodological notes

- ADE = mean **Euclidean** distance over all 12 predicted steps [m]; FDE = distance at the final step only [m]. ADE is **not** mean squared error, which would be in m^2.
- **Best-of-20** is the standard protocol for probabilistic models: 20 trajectories are sampled and the best one is kept (minimum taken per agent). Those numbers are not comparable to deterministic ones, which is why the tables are separated.
- All three models are trained and selected on the **validation** set; the test set is used exclusively for the final evaluation.
- ETH is consistently the hardest scene for every model. This is also reported in the literature (different camera angle and scene scale) and is not an artefact of this implementation.

