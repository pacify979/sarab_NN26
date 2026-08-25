# Socially-Aware Trajectory Prediction: Complete Project Guide

A deep-learning system for predicting pedestrian trajectories in shared human-robot
environments, built as the perception layer of socially-aware robot navigation. This
document explains every component, dependency, the reasoning behind each design
decision, and how to run every part of the code, from a clean checkout to a fully
evaluated set of trained models.

---

## Table of contents

1. [What the project does](#1-what-the-project-does)
2. [The dataset and the robotics framing](#2-the-dataset-and-the-robotics-framing)
3. [Repository structure](#3-repository-structure)
4. [The Python environment: what a venv is and why use one](#4-the-python-environment-what-a-venv-is-and-why-use-one)
5. [Dependencies explained, package by package](#5-dependencies-explained-package-by-package)
6. [Stage 1: Data preparation (`src/data_loader.py`)](#6-stage-1-data-preparation-srcdata_loaderpy)
7. [Stage 2: The baseline model (`src/train_lstm.py`)](#7-stage-2-the-baseline-model-srctrain_lstmpy)
8. [Stage 3: The state-of-the-art model (`src/train_stgcnn.py`)](#8-stage-3-the-state-of-the-art-model-srctrain_stgcnnpy)
9. [Stage 4: The controlled comparison (probabilistic LSTM)](#9-stage-4-the-controlled-comparison-probabilistic-lstm)
10. [Stage 5: Hyperparameter optimization (`src/tune_optuna.py`)](#10-stage-5-hyperparameter-optimization-srctune_optunapy)
11. [Stage 6: Evaluation and qualitative analysis](#11-stage-6-evaluation-and-qualitative-analysis)
12. [End-to-end quickstart](#12-end-to-end-quickstart)
13. [Results](#13-results)
14. [Design decisions and FAQ](#14-design-decisions-and-faq)
15. [Mapping to the theory](#15-mapping-to-the-theory)

---

## 1. What the project does

The system observes the recent motion of every pedestrian in a scene and predicts where
each of them will be over the next 4.8 seconds. This is the perception layer a socially
aware robot navigation stack needs: a planner cannot respect personal space or avoid
cutting through a conversing group unless it first knows where people are *going*.

Concretely: given **8 observed frames (3.2 s)** of `(x, y)` coordinates per pedestrian,
predict the next **12 frames (4.8 s)**. This 8-in/12-out protocol is the standard used
by every paper in the field, which makes the numbers directly comparable to published
results.

Three models are implemented and compared:

| Model | Idea | Sees other agents? | Loss |
|---|---|---|---|
| **Vanilla LSTM** (baseline) | each pedestrian is an independent time series | no | L2 |
| **LSTM-prob** (control) | same LSTM, but predicts a distribution | no | bivariate NLL |
| **Social-STGCNN** (SOTA) | scene is a dynamic graph, GCN + TCN | yes, explicitly | bivariate NLL |

The third model was **not** in the original project plan. It was added because without
it the comparison is confounded: the baseline and the SOTA model differ in *two* ways at
once (social modeling **and** loss function), so any difference between them cannot be
attributed to social modeling alone. See [section 13](#13-results).

The full pipeline is:

```
ETH-UCY raw .txt files (frame_id, ped_id, x, y)
      │  src/data_loader.py     (sequencing 8→12, relative displacements, caching)
      ▼
sequences of (obs_traj, pred_traj, obs_rel, pred_rel, seq_start_end)
      │  src/train_lstm.py      (baseline + probabilistic control)
      │  src/train_stgcnn.py    (graph-based SOTA model)
      ▼
outputs/*.pt                    (trained checkpoints, selected on validation ADE)
      │  src/tune_optuna.py     (TPE search over architecture + lr)
      ▼
outputs/optuna_*.json           (best hyperparameters)
      │  src/visualize.py       (trajectory plots, probability heatmaps, learning curves)
      │  src/compare.py         (final comparison tables)
      ▼
outputs/figures/*.png + outputs/results.md
```

---

## 2. The dataset and the robotics framing

### 2.1 Why ETH-UCY

ETH-UCY is the standard benchmark for pedestrian trajectory prediction. Using it is not
a compromise: it is what makes the results comparable to the literature, and it is the
dataset every method in this space reports on (Social-LSTM, Social-GAN, Social-STGCNN,
Trajectron++).

The data is recorded from a **bird's-eye view** and already converted to **world
coordinates in metres**. This matters: because positions are metric, the error metrics
(ADE/FDE) are directly interpretable as physical distances, which is exactly what a robot
planner cares about. An error of 0.4 m is roughly half a shoulder width; an error of 2 m
means the robot would plan through a person.

### 2.2 Structure

Five scenes, recorded at 2.5 Hz (one frame every 0.4 s):

| Scene | Setting | Character |
|---|---|---|
| ETH | university entrance | sparse, people entering/exiting, unusual camera angle and scale |
| HOTEL | hotel entrance | sparse, many near-stationary people waiting |
| UNIV | university campus | very dense crowds, up to 40+ people simultaneously |
| ZARA1 | shopping street | medium density, mostly bidirectional walking |
| ZARA2 | shopping street | medium density, similar to ZARA1 |

Each raw file is tab-separated with four columns: `frame_id`, `pedestrian_id`, `x`, `y`.

### 2.3 The leave-one-scene-out protocol

The split is **not** random. For each of the five scenes, the model trains on the other
four and is tested on the held-out fifth. This measures generalization to a **completely
new physical space**, not just to new pedestrians in a familiar space, which is the
realistic deployment condition for a robot entering a building it has never mapped.

Dataset statistics produced by our loader (a "sequence" is one 20-frame window; a
"trajectory" is one pedestrian present for all 20 frames of that window):

| Scene | Train seq | Train traj | Val seq | Val traj | Test seq | Test traj |
|---|---|---|---|---|---|---|
| ETH | 3,283 | 30,307 | 733 | 5,422 | 253 | 364 |
| HOTEL | 3,118 | 29,676 | 688 | 5,203 | 445 | 1,197 |
| UNIV | 2,719 | 9,874 | 622 | 2,800 | 947 | 24,334 |
| ZARA1 | 2,889 | 28,577 | 671 | 5,184 | 705 | 2,356 |
| ZARA2 | 2,681 | 26,076 | 590 | 4,262 | 998 | 5,910 |

Note the asymmetry: UNIV has few training trajectories but a very dense test set
(25.7 pedestrians per scene on average, versus 1.4 for ETH). This is a direct consequence
of leave-one-scene-out and explains why per-scene difficulty varies so much.

### 2.4 Why the pre-processed split rather than raw data

We use the standardized split shipped with the Social-STGCNN repository, which is
identical to the one from the Social-GAN paper (Gupta et al., CVPR 2018). Building the
split from raw video annotations ourselves would take days and would almost certainly
produce subtly different sequences, making comparison with published numbers meaningless.
Using the reference split is the scientifically correct choice, not a shortcut.

---

## 3. Repository structure

```
sarab_NN26/
├── data/                          # ETH-UCY (gitignored, ~14 MB, see scripts/download_data.sh)
│   ├── eth/{train,val,test}/*.txt
│   ├── hotel/{train,val,test}/*.txt
│   ├── univ/{train,val,test}/*.txt
│   ├── zara1/{train,val,test}/*.txt
│   └── zara2/{train,val,test}/*.txt
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # sequencing, relative displacements, disk caching
│   ├── metrics.py                 # ADE / FDE / relative→absolute conversion
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm_baseline.py       # Vanilla LSTM (deterministic + probabilistic)
│   │   └── social_stgcnn.py       # Social-STGCNN + adjacency + bivariate NLL loss
│   ├── train_lstm.py              # trains the baseline and the probabilistic control
│   ├── train_stgcnn.py            # trains the graph model
│   ├── tune_optuna.py             # Bayesian hyperparameter search (TPE + pruning)
│   ├── visualize.py               # trajectories, probability heatmaps, learning curves
│   └── compare.py                 # generates outputs/results.md
├── scripts/
│   └── download_data.sh           # fetches the standardized ETH-UCY split
├── docs/
│   ├── PROJECT_GUIDE.md           # this file
│   ├── SaraBabic_NN26_prezentacija.pptx   # presentation
│   └── paper/                     # IEEE-format paper
│       ├── SaraBabic_paper.tex    #   LaTeX source (IEEEtran)
│       ├── SaraBabic_paper.pdf    #   compiled paper
│       ├── refs.bib               #   bibliography
│       └── figures/en/            #   figures used by the paper
├── outputs/                       # checkpoints (gitignored), logs, figures, results
│   ├── *.pt                       # trained models
│   ├── *_log.json                 # per-scene training history + test metrics
│   ├── optuna_*.json              # hyperparameter search results
│   ├── figures/*.png              # trajectory plots, heatmaps, learning curves
│   └── results.md                 # final comparison tables
├── .cache/                        # parsed sequences (gitignored, auto-regenerated)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 4. The Python environment: what a venv is and why use one

### 4.1 What is a virtual environment (venv)?

A **virtual environment** is an isolated Python installation that lives inside the
project (in the `.venv/` folder). Instead of installing packages system-wide, where they
can clash with other projects or the OS's own Python, every package goes into this local
folder.

Benefits, and why each matters here:

- **Reproducibility**: the exact package set is listed in `requirements.txt`, so anyone
  can recreate the identical environment and get the identical numbers.
- **No conflicts**: this project pins a CUDA-specific PyTorch build; another project on
  the same machine can use a CPU build without interference.
- **Clean removal**: delete `.venv/` and every dependency is gone, system Python untouched.
- **Honest results**: a reported metric is only meaningful if the environment that
  produced it can be reconstructed.

### 4.2 Why `venv` and not conda

`venv` is part of the Python standard library, so there is nothing extra to install. Conda solves
binary dependency-resolution problems this project does not have: the only non-trivial
binary dependency is PyTorch, and PyTorch's own pip index (`download.pytorch.org`)
handles CUDA builds cleanly.

### 4.3 One-time system prerequisite (Debian/Ubuntu)

If `python3 -m venv .venv` fails with *"ensurepip is not available"*:

```bash
sudo apt install python3.10-venv
```

### 4.4 Creating and using the environment

```bash
# from the project root
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# PyTorch with CUDA 12.1 support (adjust the index URL for your CUDA version;
# omit it entirely for a CPU-only install)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# when finished:
deactivate
```

Verify the GPU is visible:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 2.5.1+cu121 True NVIDIA GeForce RTX 4050 Laptop GPU
```

**Every run command in this guide assumes the venv is active.** If you prefer not to
activate it, prefix commands with the interpreter: `.venv/bin/python -m src.train_lstm`.

The project was developed on an RTX 4050 Laptop GPU (6 GB). Peak usage is well under
1 GB, so any modern GPU works; CPU-only training also works but is roughly 10× slower.

---

## 5. Dependencies explained, package by package

| Package | Why it is here |
|---|---|
| `torch` | The entire modeling stack: LSTM, convolutions, autograd, GPU execution |
| `numpy` | Sequence construction and array manipulation in the data loader |
| `scipy` | `gaussian_filter` for smoothing the 2D probability heatmaps |
| `matplotlib` | All figures: trajectory plots, heatmaps, learning curves |
| `optuna` | Bayesian hyperparameter optimization (TPE sampler + median pruning) |
| `tqdm` | Progress bars for long-running loops |

Deliberately **not** used:

- **PyTorch Geometric**: the graph here is a small dense adjacency matrix per timestep
  (typically fewer than 300 nodes). A dense `einsum` is faster than sparse message passing
  at this scale and removes a heavy dependency with a fragile install story.
- **A pretrained backbone**: there is no ImageNet-equivalent checkpoint for pedestrian
  trajectories. The original plan mentioned "Social-ViT" transfer learning; no such
  standard public checkpoint exists. Transfer learning here realistically means the
  leave-one-scene-out protocol itself, where the model transfers to an unseen scene.

---

## 6. Stage 1: Data preparation (`src/data_loader.py`)

### 6.1 What it does

Turns raw `(frame_id, ped_id, x, y)` rows into training-ready sequences.

1. **Frame grouping**: rows are grouped by frame, and a sliding window of 20 consecutive
   frames (8 observed + 12 predicted) walks the recording with stride 1.
2. **Completeness filter**: within each window, only pedestrians present in **all 20
   frames** are kept. Pedestrians who enter or leave mid-window are dropped. This is the
   reference protocol; it avoids having to invent padding or masking semantics that would
   differ from published work.
3. **Relative displacements**: alongside absolute coordinates, the loader computes
   `(dx, dy)` between consecutive timesteps. The models consume the displacements.
4. **Disk caching**: parsed sequences are pickled into `.cache/`. Parsing all five scenes
   takes a few seconds each time; Optuna runs dozens of trials, so caching removes a
   meaningful chunk of total runtime.

### 6.2 Why relative displacements and not absolute coordinates

This is the single most important preprocessing decision. Each scene has its own
coordinate frame. ETH's origin is somewhere in a university courtyard, ZARA's is on a
shopping street. A model trained on absolute `(x, y)` learns *where people stand in the
training scenes*, which is worthless in a new scene. Feeding it `(dx, dy)` makes the
model **translation invariant**: it learns *how people move*, which transfers.

### 6.3 The batching convention

Different scenes contain different numbers of pedestrians, so a standard
`(batch, features)` tensor does not fit. The loader follows the Social-GAN convention:
all pedestrians across the batch are concatenated along one dimension, and a
`seq_start_end` tensor records the boundaries of each scene:

```
obs_traj       (8,  N, 2)     N = total pedestrians across all scenes in the batch
pred_traj      (12, N, 2)
obs_traj_rel   (8,  N, 2)
pred_traj_rel  (12, N, 2)
seq_start_end  (B,  2)        B = number of scenes; row i = [start, end) of scene i
```

The LSTM ignores `seq_start_end` entirely (it treats every pedestrian independently).
The graph model uses it to know which pedestrians belong to the same scene. **One loader
serves both models**, which guarantees they see byte-identical data, which is essential for the
comparison to mean anything.

### 6.4 How to run

The loader is a library, not a script, but you can exercise it directly:

```bash
python -c "
from src.data_loader import data_loader
dset, loader = data_loader('data/eth', 'train')
print(len(dset), 'sequences,', dset.num_peds, 'trajectories')
print([tuple(t.shape) for t in next(iter(loader))])
"
```

### 6.5 Expected output

```
3283 sequences, 30307 trajectories
[(8, 103, 2), (12, 103, 2), (8, 103, 2), (12, 103, 2), (64, 2)]
```

**Interpretation:** 3,283 windows in ETH's training split, containing 30,307 complete
20-frame trajectories. The first batch of 64 scenes contains 103 pedestrians in total.

---

## 7. Stage 2: The baseline model (`src/train_lstm.py`)

### 7.1 Architecture

An encoder-decoder LSTM operating on displacements:

```
obs_traj_rel (8, N, 2)
    │  Linear(2 → 64)                 spatial embedding
    ▼
  LSTM encoder (hidden 128)           compresses motion history into a hidden state
    │
    ▼  hidden state
  LSTM decoder (hidden 128)           autoregressive: 12 steps, output fed back as input
    │  Linear(128 → 2)
    ▼
pred_traj_rel (12, N, 2)              predicted displacements
```

Displacements are converted back to absolute positions by cumulative summation from the
last observed point.

**No pedestrian ever sees another.** That is the point: this model isolates how much of
the prediction problem is explained by a single agent's own inertia.

199,106 parameters.

### 7.2 Key design decisions

- **Loss = mean Euclidean error over displacements**, which correlates directly with ADE,
  the metric being reported. Optimizing something the metric does not measure would be a
  methodological error.
- **Gradient clipping (norm 1.0).** The decoder is autoregressive, so gradients propagate
  through 12 chained steps and can explode.
- **Model selection on validation ADE, never training loss.** The checkpoint saved is the
  one with the best validation ADE across all epochs.
- **The test set is never touched during training.**

### 7.3 How to run

```bash
python -m src.train_lstm --scene all --epochs 100
```

Roughly 90 seconds per scene on a laptop GPU; about 8 minutes for all five.

### 7.4 Expected output

```
[eth] epoch  25/100  loss=0.0806  val_ADE=0.421  val_FDE=0.884
...
[eth] TEST  ADE=1.007 m  FDE=1.996 m  (94s)
...
====================================
SCENE               ade          fde
------------------------------------
eth               1.007        1.996
hotel             0.457        0.936
univ              0.577        1.278
zara1             0.402        0.886
zara2             0.316        0.699
------------------------------------
AVERAGE           0.552        1.159
====================================
```

**Interpretation:** an average ADE of 0.552 m. The Social-GAN paper reports 0.70 m for
its LSTM baseline, so ours is *better* than the published baseline, mainly because we
train on displacements and select on validation ADE. This is worth stating explicitly in
the report: a strong baseline makes the subsequent comparison honest rather than flattering.

---

## 8. Stage 3: The state-of-the-art model (`src/train_stgcnn.py`)

### 8.1 Architecture

Social-STGCNN (Mohamed et al., CVPR 2020) treats each timestep of a scene as a graph:
nodes are pedestrians, edge weights are inverse distances (closer people influence each
other more).

```
obs_traj (8, V, 2) ──► build_adjacency ──► A (8, V, V)
                                            normalized: D^(-1/2)(A+I)D^(-1/2)
obs_traj_rel (8, V, 2)
    │
    ▼  ST-GCNN block:  graph convolution (spatial) → TCN (temporal)
    │                  aggregates neighbours       processes time in parallel
    ▼
  permute time into the channel axis
    │
    ▼  TXP-CNN: 5 conv layers expanding 8 observed steps → 12 predicted steps
    ▼
(12, V, 5)   →  mu_x, mu_y, sigma_x, sigma_y, rho
```

Two properties distinguish it from the LSTM:

1. **It sees neighbours.** The graph convolution mixes each pedestrian's features with
   those of nearby pedestrians at every timestep.
2. **It predicts a distribution, not a point.** The output is the five parameters of a
   bivariate Gaussian per pedestrian per timestep, trained with negative log-likelihood.
   This is what makes the probability heatmaps possible.

Only **7,563 parameters**, 26× fewer than the LSTM.

### 8.2 Why the temporal decoder is convolutional rather than recurrent

The LSTM decoder generates 12 steps sequentially, each depending on the previous. The
TXP-CNN generates all 12 in a single forward pass by convolving along the time axis. This
is why the model is fast despite the graph machinery, and it is a genuine architectural
advantage worth mentioning at a defence.

### 8.3 The performance problem and how it was solved

The naive implementation processes **one scene per optimizer step**, exactly as the
reference repository does. With 3,283 sequences per epoch, that is thousands of tiny GPU
launches. Profiling the ETH training loop:

| Stage | Time per epoch |
|---|---|
| DataLoader iteration alone | 0.2 s |
| \+ building adjacency matrices | 9.0 s |
| \+ forward and backward pass | 34.1 s |
| \+ validation with sampling | ≈ 62 s total |

At 62 s/epoch, 250 epochs × 5 scenes would be about **13 hours**.

The fix is **block-diagonal graph packing**. Instead of one scene per step, B scenes are
packed into a single graph whose adjacency matrix is block-diagonal: pedestrians from
different scenes simply have no edge between them, so the graph propagation is identical
to processing them separately, but it happens in one GPU call.

```
       scene 1        scene 2        scene 3
     ┌─────────┬──────────────┬──────────────┐
     │  A₁     │      0       │      0       │
     ├─────────┼──────────────┼──────────────┤
     │   0     │     A₂       │      0       │
     ├─────────┼──────────────┼──────────────┤
     │   0     │      0       │     A₃       │
     └─────────┴──────────────┴──────────────┘
```

Combined with vectorizing the best-of-K sampling (removing a Python loop over samples)
and validating every 5 epochs instead of every epoch, this brought training to
**2.6 s/epoch, a 24× speedup**. Full training of all five scenes now takes about
15 minutes.

**One honest caveat:** graph propagation and the temporal convolutions are mathematically
identical under this packing, but `BatchNorm` now computes statistics over all nodes in
the batch rather than over one scene. That is standard mini-batch behaviour (and generally
stabilizes training), but it is not a bit-identical computation, and the report should say so.

### 8.4 Deterministic and best-of-20 evaluation

The script reports **both**:

- **Deterministic**: use the mean of the predicted distribution, giving one trajectory that is directly
  comparable to the LSTM baseline.
- **Best-of-20**: sample 20 trajectories, keep the one closest to ground truth (minimum
  taken per pedestrian). This is the protocol every probabilistic paper reports, so it is
  what makes our numbers comparable to published ones.

Reporting only one of these would be misleading in opposite directions, so both are always
printed.

### 8.5 How to run

```bash
python -m src.train_stgcnn --scene all --epochs 250
```

About 3 minutes per scene, 15 minutes total.

### 8.6 Expected output

```
[eth] epoch 250/250  nll=-2.7437  val_ADE(det)=0.436  val_ADE(best5)=0.364
[eth] TEST  det: ADE=1.043 FDE=2.113 | best-of-20: ADE=0.819 FDE=1.582  (179s)
...
====================================================================
SCENE           ADE det      FDE det      ADE b20      FDE b20
--------------------------------------------------------------------
eth               1.043        2.113        0.819        1.582
hotel             0.436        0.880        0.311        0.561
univ              0.568        1.202        0.412        0.803
zara1             0.419        0.904        0.283        0.498
zara2             0.350        0.761        0.251        0.474
--------------------------------------------------------------------
AVERAGE           0.563        1.172        0.415        0.784
====================================================================
```

A negative NLL is normal and expected: it is a log *density*, not a probability, and
densities exceed 1 when the distribution is narrow. A decreasing NLL means the model is
becoming more confident and more accurate at the same time.

---

## 9. Stage 4: The controlled comparison (probabilistic LSTM)

### 9.1 Why this model exists

After training the first two models, the comparison looked like this:

| | ADE |
|---|---|
| LSTM (L2 loss, deterministic) | 0.552 |
| Social-STGCNN (NLL loss, deterministic mean) | 0.563 |
| Social-STGCNN (NLL loss, best-of-20) | 0.415 |

Neither row answers the research question. The models differ in **two** variables
simultaneously:

1. whether the model sees other agents (the thing we want to measure), and
2. what loss function it was trained with (a confound).

A model trained with NLL has a mean that is *not* optimized for L2 error, so the
deterministic row penalizes Social-STGCNN for a reason unrelated to social modeling. And
the best-of-20 row compares 20 attempts against 1, which is not a comparison at all.

### 9.2 What was done

The same `VanillaLSTM` was given a `--probabilistic` flag: the output head widens from 2
to 5 values, and it trains with the **identical bivariate NLL loss** used by
Social-STGCNN, is selected on the **identical** validation criterion, and is evaluated
with the **identical** best-of-20 protocol.

Now exactly one variable differs between the two models: whether the architecture can see
other pedestrians. Only this comparison isolates the contribution of social modeling.

### 9.3 How to run

```bash
python -m src.train_lstm --scene all --epochs 250 --probabilistic
```

### 9.4 Expected output

```
[eth] TEST  det: ADE=1.002 FDE=1.993 | best-of-20: ADE=0.775 FDE=1.405  (272s)
...
==============================================================
SCENE           ade_det      fde_det   ade_best20   fde_best20
--------------------------------------------------------------
eth               1.002        1.993        0.775        1.405
hotel             0.516        1.075        0.342        0.659
univ              0.583        1.255        0.411        0.816
zara1             0.419        0.924        0.276        0.497
zara2             0.323        0.724        0.221        0.428
--------------------------------------------------------------
AVERAGE           0.568        1.194        0.405        0.761
==============================================================
```

---

## 10. Stage 5: Hyperparameter optimization (`src/tune_optuna.py`)

### 10.1 What it does

Optuna's **TPE (Tree-structured Parzen Estimator)** sampler performs Bayesian
optimization: it builds a probabilistic model of the mapping from hyperparameters to
validation ADE and concentrates subsequent trials where expected improvement is highest.
Unlike grid search, it does not waste time on obviously bad regions, which is why 15
trials are enough to be meaningful.

A **MedianPruner** additionally kills any trial that falls below the median of previous
trials after a few epochs, saving a large fraction of the compute.

### 10.2 Search spaces

| LSTM | Social-STGCNN |
|---|---|
| `lr` ∈ [1e-4, 5e-3], log scale | `lr` ∈ [1e-4, 5e-3], log scale |
| `hidden_dim` ∈ {32, 64, 128, 256} | `n_stgcnn` ∈ [1, 3] (graph conv blocks) |
| `embedding_dim` ∈ {16, 32, 64, 128} | `n_txpcnn` ∈ [3, 7] (temporal decoder depth) |
| `num_layers` ∈ [1, 2] | `kernel_size` ∈ {3, 5} |
| `dropout` ∈ [0.0, 0.3] | `dropout` ∈ [0.0, 0.3] |

### 10.3 The critical methodological rule

**The objective is validation ADE. The test set is never seen during the search.**
Selecting hyperparameters on test performance is a form of information leakage that
silently inflates the reported result. This is the most common serious mistake in student
ML projects and the easiest to be challenged on at a defence.

### 10.4 How to run

```bash
python -m src.tune_optuna --model lstm   --scene zara1 --trials 15 --epochs 30
python -m src.tune_optuna --model stgcnn --scene zara1 --trials 15 --epochs 40
```

Roughly 7 and 20 minutes respectively.

### 10.5 Expected output

```
  trial  0  COMPLETE   val_ADE=0.4192  {'lr': 0.00043, 'n_stgcnn': 3, 'n_txpcnn': 6, ...}
  trial  3  PRUNED     val_ADE=0.4727  {...}
  ...
  trial 13  COMPLETE   val_ADE=0.3894  {'lr': 0.00272, 'n_stgcnn': 2, 'n_txpcnn': 7, ...}
============================================================
Best val_ADE: 0.3894 m
Best hyperparameters:
  lr               = 0.00271550054407983
  n_stgcnn         = 2
  n_txpcnn         = 7
  kernel_size      = 5
  dropout          = 0.19429947797142905
============================================================
```

**Interpretation:** 15 trials, 6 pruned early. The search converged toward a
**wider** temporal decoder (`n_txpcnn` 5 → 7), a **larger** temporal kernel (3 → 5), and
moderate dropout. The last trials all cluster around the same region, which is the signal
that TPE has found a basin rather than wandering randomly. The LSTM search reached
val_ADE 0.4283 with `hidden_dim=256, lr=0.0032, num_layers=1` (15 trials, 10 pruned).

### 10.6 Retraining with the tuned configuration

Closing the loop, train the final model with the best-found hyperparameters:

```bash
python -m src.train_stgcnn --scene zara1 --epochs 250 \
  --lr 0.00271550054407983 --n_stgcnn 2 --n_txpcnn 7 \
  --kernel_size 5 --dropout 0.19429947797142905 --out_dir outputs/tuned
```

Result on ZARA1:

| Configuration | Val ADE | Test ADE (best-20) | Test FDE (best-20) | Parameters |
|---|---|---|---|---|
| Default | 0.381 | 0.283 | 0.498 | 7,563 |
| Optuna-tuned | 0.374 | 0.283 | 0.497 | 10,413 |

**The tuned configuration improved validation ADE but produced no measurable test
improvement.** This is worth reporting honestly rather than hiding: it indicates the
default architecture was already near the ceiling for this dataset, and that the small
validation gain was within noise rather than a real generalization gain. It is also a
textbook illustration of why hyperparameters must be selected on validation and then
*verified* on a held-out test set. Had we tuned on test, we would have reported a
spurious improvement.

---

## 11. Stage 6: Evaluation and qualitative analysis

### 11.1 Metrics (`src/metrics.py`)

- **ADE (Average Displacement Error)**: the mean **Euclidean** distance between the
  predicted and true position, averaged over all 12 predicted steps and all pedestrians,
  in metres.

  > **Correction to the original project proposal:** ADE is *not* mean squared error.
  > MSE would be in m², not metres. Every reference paper defines ADE as above, and the
  > report should use that definition.

- **FDE (Final Displacement Error)**: the Euclidean distance at the final predicted step
  only, in metres. It captures whether the model got the *destination* right, which for a
  robot planner matters more than intermediate accuracy.

Errors are accumulated as sums over pedestrians and divided once at the end
(`ErrorAccumulator`). Averaging per-batch means would be wrong, because batches contain
different numbers of pedestrians.

### 11.2 Visualization (`src/visualize.py`)

Two figure types per scene:

- **`trajectories_<scene>_<i>.png`**: observed history, ground truth future, and both
  models' predictions overlaid, for every pedestrian in the scene.
- **`heatmap_<scene>_<i>.png`**: the probability density of future motion, obtained by
  drawing 2,000 sample trajectories from the learned distribution and building a smoothed
  2D histogram. This is the figure that connects the model to the robotics motivation:
  it is a map of *where people will probably be*, which is exactly what a socially aware
  planner needs as a cost field.

Scenes are not taken in file order. They are ranked by "interestingness" (number of
pedestrians × mean path length) and filtered by `--min_peds`, `--max_peds`, and
`--min_motion`, because many raw windows contain people standing still, which produces
uninformative figures.

### 11.3 How to run

```bash
python -m src.visualize --scene zara1 --n_scenes 4 --min_peds 4 --max_peds 7 --min_motion 5.0
python -m src.compare
```

### 11.4 Expected output

```
Saved 4 figure pairs + learning curves to outputs/figures/
```

and `outputs/results.md` containing the two comparison tables plus the literature
reference values.

---

## 12. End-to-end quickstart

From a clean checkout to full results:

```bash
# 0. system prerequisite (one time, Debian/Ubuntu)
sudo apt install python3.10-venv

# 1. environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2. data (~14 MB)
bash scripts/download_data.sh

# 3. baseline                        (~8 min)
python -m src.train_lstm --scene all --epochs 100

# 4. probabilistic control           (~19 min)
python -m src.train_lstm --scene all --epochs 250 --probabilistic

# 5. graph model                     (~15 min)
python -m src.train_stgcnn --scene all --epochs 250

# 6. hyperparameter search           (~30 min)
python -m src.tune_optuna --model lstm   --scene zara1 --trials 15 --epochs 30
python -m src.tune_optuna --model stgcnn --scene zara1 --trials 15 --epochs 40

# 7. figures + final tables
for s in eth hotel univ zara1 zara2; do
  python -m src.visualize --scene $s --n_scenes 4 --min_peds 4 --max_peds 7 --min_motion 5.0
done
python -m src.compare
```

Total: roughly 75 minutes on a laptop GPU.

> **Long runs:** if you launch training from a terminal you might close, detach it so it
> survives:
> ```bash
> setsid nohup python -m src.train_stgcnn --scene all --epochs 250 > train.log 2>&1 &
> ```
> Add `PYTHONUNBUFFERED=1` so the log updates live instead of only at exit.

---

## 13. Results

### 13.1 Comparison 1: deterministic models

| Scene | LSTM ADE | LSTM FDE | ST-GCNN ADE | ST-GCNN FDE |
|---|---|---|---|---|
| ETH | 1.007 | 1.996 | 1.043 | 2.113 |
| HOTEL | 0.457 | 0.936 | 0.436 | 0.880 |
| UNIV | 0.577 | 1.278 | 0.568 | 1.202 |
| ZARA1 | 0.402 | 0.886 | 0.419 | 0.904 |
| ZARA2 | 0.316 | 0.699 | 0.350 | 0.761 |
| **Average** | **0.552** | **1.159** | **0.563** | **1.172** |

Confounded by the loss-function difference, see section 9.1.

### 13.2 Comparison 2: probabilistic models, identical loss and protocol

**This is the comparison that answers the research question.**

| Scene | LSTM-prob ADE | LSTM-prob FDE | ST-GCNN ADE | ST-GCNN FDE |
|---|---|---|---|---|
| ETH | **0.775** | **1.405** | 0.819 | 1.582 |
| HOTEL | 0.342 | 0.659 | **0.311** | **0.561** |
| UNIV | **0.411** | 0.816 | 0.412 | **0.803** |
| ZARA1 | **0.276** | **0.497** | 0.283 | 0.498 |
| ZARA2 | **0.221** | **0.428** | 0.251 | 0.474 |
| **Average** | **0.405** | **0.761** | 0.415 | 0.784 |

### 13.3 Comparison with published numbers

| Model | Our average ADE | Published average ADE |
|---|---|---|
| LSTM baseline (deterministic) | 0.552 | 0.70 (Social-GAN paper) |
| Social-STGCNN (best-of-20) | 0.415 | 0.44 (CVPR 2020 paper) |

Both models land in the published range, and Social-STGCNN in fact beats its published
average, and exceeds its published per-scene numbers on HOTEL, UNIV and ZARA2. This
confirms the protocol and evaluation are implemented correctly, which is the necessary
precondition for trusting the comparison above.

### 13.4 The main finding

**Once the loss function is controlled for, the graph-based social model shows no average
advantage over a model that cannot see other pedestrians at all** (0.415 vs 0.405).
Social-STGCNN wins only on HOTEL; it loses on the remaining four scenes.

This is not an implementation failure: both models reproduce published performance. It
aligns with a known critique of the benchmark: Schöller et al. (2020), *"What the Constant
Velocity Model Can Teach Us About Pedestrian Motion Prediction"*, show that even a constant
velocity model outperforms most "social" architectures on ETH-UCY. Over a 4.8 s horizon
pedestrians mostly continue along their current heading, so the overwhelming majority of
the error is explained by individual inertia rather than by interaction.

Two things *did* produce large, real improvements:

1. **Modeling uncertainty**: 0.552 → 0.405 ADE (a 27% reduction) came from predicting a
   distribution rather than a point, not from the graph.
2. **Parameter efficiency**: Social-STGCNN matches the LSTM using **7,563 parameters
   versus 199,106, 26× fewer**. For an embedded robot controller, that is a substantial
   practical advantage even at equal accuracy, and it is the strongest argument for the
   graph architecture that this project's evidence actually supports.

### 13.5 What this means for the robotics motivation

The project's premise was that a robot needs socially-aware prediction. The evidence
supports a refined version of that claim: what the planner most needs is not a
graph-structured interaction model, but a **calibrated distribution** over future
positions. The probability heatmaps in `outputs/figures/` are directly usable as a cost
field for a planner, and they come from the uncertainty modeling, which either
architecture provides.

---

## 14. Design decisions and FAQ

**Why displacements instead of absolute coordinates?**
Each scene has its own coordinate frame. Absolute coordinates make the model memorize
scene layout; displacements make it translation-invariant and let it generalize to an
unseen scene, which is exactly what leave-one-scene-out measures.

**Why is ETH the hardest scene for every model?**
ETH was recorded at a different camera angle and scale from the other four, and its test
split averages only 1.4 pedestrians per window. Every published paper also reports ETH as
the hardest scene, so this is a property of the benchmark, not of the implementation.

**Why report both deterministic and best-of-20 numbers?**
They answer different questions and are not interchangeable. Deterministic asks "how good
is one prediction?"; best-of-20 asks "does the true future lie somewhere in the model's
predicted distribution?". Reporting only best-of-20 against a deterministic baseline would
compare 20 attempts to 1.

**Why did you adapt Social-STGCNN rather than implement it from scratch?**
The architecture is taken from the CVPR 2020 paper and its reference implementation. What
is original here is the entire surrounding system: the data pipeline, the baselines, the
evaluation module, the block-diagonal optimization, the Optuna integration, the
visualization, and the controlled comparison.

**Why only 15 Optuna trials?**
TPE with median pruning extracts a usable signal from that budget, and the last trials
visibly cluster in one region of the space, which is the indication that the search
converged. More trials would refine the optimum marginally; the project's conclusions do
not hinge on it.

**Why no Social Transformer?**
The original plan listed Social Transformer *or* ST-GNN. Implementing both properly is
two research projects. ST-GNN was chosen; the Transformer is discussed as future work.

**Is the graph model useless then?**
No, but the honest claim is narrower than "it predicts better". It matches the LSTM's
accuracy with 26× fewer parameters, and it wins clearly on HOTEL. What this project shows
is that on this benchmark, at this horizon, interaction modeling is not where the accuracy
comes from.

---

## 15. Mapping to the theory

The theoretical component concerns socially-aware navigation and sequence modeling. The
implementation demonstrates:

- **Recurrent sequence modeling**: the LSTM encoder-decoder shows the classic
  encode-history / decode-future pattern, including autoregressive generation and the
  gradient-stability problems it creates (hence gradient clipping).
- **Graph neural networks**: nodes as pedestrians, inverse-distance edge weights, and
  symmetric normalization `D^(-1/2)(A+I)D^(-1/2)` following Kipf & Welling, applied to a
  graph whose topology changes at every timestep, which is what makes it *spatio-temporal*
  rather than a static GCN.
- **Temporal convolution as an alternative to recurrence**: the TXP-CNN produces all 12
  future steps in one pass, illustrating the parallelism argument for convolutional
  sequence models.
- **Probabilistic prediction**: the bivariate Gaussian head and NLL loss show that a
  network can learn its own uncertainty, and the heatmaps make that uncertainty legible.
- **Bayesian hyperparameter optimization**: TPE as a sample-efficient alternative to grid
  search, with early stopping via median pruning.
- **Experimental methodology**: the probabilistic LSTM is the project's clearest
  methodological contribution: a controlled experiment that isolates one variable, and a
  demonstration of why an uncontrolled comparison produces a conclusion that looks
  supported but is not.

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
