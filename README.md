# Socially-Aware Trajectory Prediction in Dynamic Environments

Neural networks course project: predicting future pedestrian trajectories on the ETH-UCY
dataset, as the perception layer of socially-aware robot navigation.

Three models are compared:

| Model | Idea | Sees other agents? | Loss |
|---|---|---|---|
| **Vanilla LSTM** (baseline) | each pedestrian is an independent time series | no | L2 |
| **LSTM-prob** (control) | same LSTM, but predicts a distribution | no | bivariate NLL |
| **Social-STGCNN** (SOTA) | scene is a dynamic graph, GCN + TCN | yes, explicitly | bivariate NLL |

The third model was not part of the original plan. It was added because without it the
contribution of social modelling cannot be separated from the contribution of the loss
function (see the main finding below).

The protocol is the field standard: from **8 observed frames (3.2 s)** the model predicts
the next **12 frames (4.8 s)**, evaluated **leave-one-scene-out** (trained on four scenes,
tested on a fifth the model has never seen).

---

## Project structure

```
sarab_NN26/
├── data/                       ETH-UCY (eth, hotel, univ, zara1, zara2), train/val/test
├── src/
│   ├── data_loader.py          loading and sequencing of trajectories (8 -> 12)
│   ├── metrics.py              ADE / FDE / relative -> absolute conversion
│   ├── models/
│   │   ├── lstm_baseline.py    Vanilla LSTM (encoder-decoder, deterministic + probabilistic)
│   │   └── social_stgcnn.py    Social-STGCNN + bivariate NLL loss
│   ├── train_lstm.py           trains the baseline and the probabilistic control
│   ├── train_stgcnn.py         trains the graph model
│   ├── tune_optuna.py          Bayesian hyperparameter optimisation (TPE)
│   ├── visualize.py            trajectories + probability heatmaps + learning curves
│   └── compare.py              generates the final comparison tables (Markdown)
├── docs/
│   ├── PROJECT_GUIDE.md        full project guide: every component and design decision
│   ├── SaraBabic_NN26_prezentacija.pptx   defence presentation
│   └── paper/                  IEEE-format paper (SaraBabic_paper.tex/.pdf) + refs.bib
└── outputs/                    checkpoints, logs, figures, results.md
```

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install -r requirements.txt
```

The `--index-url` matters: without it pip installs the CPU-only build and training is
roughly 10x slower.

The dataset is fetched by a script (it is not in git - 14 MB of third-party data):

```bash
bash scripts/download_data.sh
```

This uses the standardised split from the Social-STGCNN repository, identical to the one
from the Social-GAN paper, which is what makes the results directly comparable to
published numbers.

## Running

```bash
# 1) baseline on all 5 scenes
.venv/bin/python -m src.train_lstm --scene all --epochs 100

# 2) probabilistic control on all 5 scenes (same 250-epoch budget as the graph model)
.venv/bin/python -m src.train_lstm --scene all --epochs 250 --probabilistic

# 3) graph model on all 5 scenes
.venv/bin/python -m src.train_stgcnn --scene all --epochs 250

# 4) automatic hyperparameter calibration
.venv/bin/python -m src.tune_optuna --model stgcnn --scene zara1 --trials 15 --epochs 40

# 5) qualitative analysis (figures in outputs/figures/)
.venv/bin/python -m src.visualize --scene zara1 --n_scenes 6

# 6) final comparison tables -> outputs/results.md
.venv/bin/python -m src.compare
```

For long runs launched from a terminal you may close, detach the process so it survives:

```bash
setsid nohup .venv/bin/python -m src.train_stgcnn --scene all --epochs 250 > train.log 2>&1 &
```

Prefix with `PYTHONUNBUFFERED=1` so the log updates live rather than only at exit.

---

## Main finding

The complete tables are in `outputs/results.md`. Summarised (average over 5 scenes, ADE in
metres):

| Comparison | LSTM | ST-GCNN | Difference |
|---|---|---|---|
| Deterministic (L2 vs NLL mean) | **0.552** | 0.563 | -2.1% |
| Probabilistic, best-of-20 (identical NLL) | **0.405** | 0.415 | -2.5% |

**Once the loss function is controlled for, the graph-based model shows no average
advantage over a model that cannot see other agents at all.** ST-GCNN wins only on HOTEL
(0.311 vs 0.342); it loses on the other four scenes.

This is not an implementation failure - both models land within the published range. The
finding matches a known critique in the literature: Schöller et al. (2020) show that even a
plain constant velocity model outperforms most "social" models on ETH-UCY. The reason is
that over a 4.8 s horizon pedestrians mostly continue in a straight line, so most of the
error is explained by motion inertia rather than by interaction.

Two things did produce real improvements:

1. **Uncertainty modelling**: 0.552 -> 0.405 ADE (a 27% reduction), obtained by predicting
   a distribution instead of a point - not by the graph.
2. **Parameter efficiency**: Social-STGCNN matches the LSTM using **7,563 versus 199,106
   parameters, 26x fewer**. For an embedded robot controller that is a substantial
   practical advantage even at equal accuracy.

**Practical implication for robotics:** what a planner most needs is a calibrated
distribution over future positions, and the probability heatmaps in `outputs/figures/` are
directly usable as a cost field. That comes from uncertainty modelling, which either
architecture provides.

---

## Key design decisions

**1. The models operate on relative displacements, not absolute coordinates.**
Every scene has its own coordinate frame, so a model fed absolute coordinates learns
*where people stand in that scene* instead of *how people move*. Working with displacements
(dx, dy) gives translation invariance and substantially better generalisation to an unseen
scene.

**2. ADE is not mean squared error.**
ADE is the mean **Euclidean distance** between the predicted and the true position,
averaged over all 12 steps and all agents - which is why it is expressed in metres (MSE
would be in m²). This is the definition used in every reference paper.

**3. Deterministic is compared with deterministic.**
Social-STGCNN predicts a **distribution**, so the literature reports its ADE as
*best-of-20* (20 paths are sampled and the best is kept). The LSTM produces one path.
Comparing those two numbers directly would be incorrect, so **both** variants are computed
for ST-GCNN: deterministic (the mean of the distribution) for a fair comparison against the
baseline, and best-of-20 for comparison against published numbers.

**4. Block-diagonal graph packing.**
A naive implementation processes one scene per optimiser step, which over 3,283 sequences
means thousands of tiny GPU launches (~62 s per epoch). Instead, several scenes are packed
into a single block-diagonal adjacency matrix - pedestrians from different scenes have no
edge between them, so graph propagation is identical, but training is **~24x faster**
(~2.6 s per epoch). The one real difference is that BatchNorm computes statistics over the
whole batch rather than over a single scene, which is standard mini-batch behaviour.

**5. Optuna never sees the test set.**
Validation ADE is optimised. Selecting hyperparameters on the test set would be information
leakage and the reported result would not be honest.

---

## What is adapted and what is original

Stated explicitly for academic correctness.

**Adapted from published work / third-party repositories:**

| Component | Source | What exactly was taken |
|---|---|---|
| Social-STGCNN architecture | Mohamed et al., CVPR 2020; reference implementation `github.com/abduallahmohamed/Social-STGCNN` (MIT licence) | The architectural design: graph convolution + TCN blocks, the TXP-CNN temporal decoder, inverse-distance adjacency, and the 5-parameter Gaussian output head. Re-implemented in `src/models/social_stgcnn.py` following the paper and that repository. |
| ETH-UCY dataset split | Distributed inside the Social-STGCNN repository; originally from Social-GAN, Gupta et al., CVPR 2018 (`github.com/agrimgupta92/sgan`) | The pre-processed `train/val/test` text files and the leave-one-scene-out partitioning. Downloaded verbatim by `scripts/download_data.sh`; not modified. |
| Sequencing protocol | Social-GAN, Gupta et al., CVPR 2018 | The 8-in/12-out windowing convention and the rule that only pedestrians present in all 20 frames are retained. The loader code in `src/data_loader.py` is our own, but follows this protocol so the numbers stay comparable. |
| Batching convention | Social-GAN | Concatenating all pedestrians along one dimension with a `seq_start_end` boundary tensor. |
| Bivariate Gaussian NLL loss | Social-LSTM (Alahi et al., 2016) and Social-STGCNN | The mathematical formulation. The implementation, including the numerical-stability clamping, is ours. |
| Underlying data | ETH sequences from Pellegrini et al., ICCV 2009; UCY sequences from Lerner et al., 2007 | The original recordings and annotations. |

**Implemented independently:**
- The complete data pipeline and loader, with disk caching and an O(1) frame index
- The Vanilla LSTM baseline (encoder-decoder over displacements)
- The probabilistic LSTM control -- the controlled experiment that isolates the effect of
  social modelling; this is the project's original methodological contribution
- The evaluation module: ADE/FDE with correct per-agent aggregation, deterministic and
  best-of-K variants
- The block-diagonal batching optimisation (24x speedup)
- The Optuna integration with pruning
- All visualisation, including the sampled probability heatmaps
- The comparative analysis against published reference values

### Status of the Social-STGCNN implementation

To be precise about what "adapted" means here: **no code and no weights were copied from
the reference repository.** `src/models/social_stgcnn.py` is a re-implementation written
against the paper, and every model in this project was trained from random initialisation
on our own pipeline. There was no fine-tuning of a released checkpoint -- none is published
for this model. Concrete differences from the reference implementation:

| Aspect | Reference implementation | This project |
|---|---|---|
| Adjacency normalisation | normalised graph **Laplacian**, `nx.normalized_laplacian_matrix`, i.e. `L = I - D^-1/2 A D^-1/2` | normalised **adjacency** `D^-1/2 (A+I) D^-1/2` (standard GCN rule, Kipf & Welling). Since `L = I - Â`, these are different operators |
| Graph construction | precomputed per sequence in NumPy/NetworkX inside the data loader | built on the GPU at forward time with `torch.cdist` |
| Batching | one scene per optimiser step, gradients accumulated over 128 steps | several scenes packed into one block-diagonal graph, real mini-batches |
| Optimiser | SGD, `StepLR` with gamma 0.2 | Adam, `StepLR` with gamma 0.5 |
| Dropout | not exposed | exposed as a hyperparameter and searched by Optuna |
| NLL loss | no numerical guards | sigma, rho and the log argument are clamped for stability |
| Evaluation | best-of-20 only | both deterministic and best-of-20, so the comparison against the baseline is fair |

The deviation in adjacency normalisation is documented in the paper as well. Our results
reproduce (and on three scenes exceed) the published figures, so the choice is defensible,
but it should be disclosed rather than glossed over.

---

## Documentation

| Document | Language | Content |
|---|---|---|
| `docs/PROJECT_GUIDE.md` | English | Full guide: every component, dependency and design decision, with expected outputs |
| `docs/paper/SaraBabic_paper.pdf` | English | IEEE-format paper |
| `docs/SaraBabic_NN26_prezentacija.pptx` | Serbian | Defence presentation, 16 slides |
| `outputs/results.md` | English | Generated comparison tables |

To rebuild the paper (TeX Live is installed locally via TinyTeX, no root required):

```bash
export PATH=$HOME/.TinyTeX/bin/x86_64-linux:$PATH
cd docs/paper
pdflatex SaraBabic_paper && bibtex SaraBabic_paper && pdflatex SaraBabic_paper && pdflatex SaraBabic_paper
```

Four passes are needed to resolve cross-references and the bibliography.

---

## References

1. Gupta et al., *Social GAN: Socially Acceptable Trajectories with Generative Adversarial
   Networks*, CVPR 2018.
2. Mohamed et al., *Social-STGCNN: A Social Spatio-Temporal Graph Convolutional Neural
   Network for Human Trajectory Prediction*, CVPR 2020.
3. Alahi et al., *Social LSTM: Human Trajectory Prediction in Crowded Spaces*, CVPR 2016.
4. Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional Networks*,
   ICLR 2017.
5. Schöller et al., *What the Constant Velocity Model Can Teach Us About Pedestrian Motion
   Prediction*, IEEE RA-L 2020.
6. Akiba et al., *Optuna: A Next-generation Hyperparameter Optimization Framework*, KDD 2019.
7. Hochreiter & Schmidhuber, *Long Short-Term Memory*, Neural Computation, 1997.
8. Pellegrini et al., *You'll Never Walk Alone: Modeling Social Behavior for Multi-Target
   Tracking*, ICCV 2009.
9. Lerner et al., *Crowds by Example*, Computer Graphics Forum, 2007.
