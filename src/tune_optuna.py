"""
Automatska kalibracija hiperparametara pomocu Optuna (Bayesian optimization).

Primer:
    python -m src.tune_optuna --model stgcnn --scene zara1 --trials 15 --epochs 40
    python -m src.tune_optuna --model lstm   --scene zara1 --trials 15 --epochs 30

Kako radi (za izvestaj):
Optuna koristi TPE (Tree-structured Parzen Estimator) -- Bayesovsku metodu koja
gradi verovatnosni model odnosa hiperparametri -> rezultat i sledeci trial bira
tamo gde je ocekivano poboljsanje najvece. Za razliku od grid search-a, ne trosi
vreme na ocigledno lose kombinacije, pa je efikasan i sa svega 15-20 pokusaja.

Dodatno je ukljucen MedianPruner: trial koji je posle nekoliko epoha losiji od
medijane dotadasnjih trial-ova se prekida ranije, sto stedi jos vremena.

VAZNO: optimizuje se VALIDACIONI ADE. Test skup se ne dodiruje tokom pretrage --
inace bi izbor hiperparametara bio oblik curenja informacija iz test skupa.
"""

import argparse
import json
import os

import optuna
import torch
import torch.nn as nn

from src.metrics import relative_to_abs
from src.models.lstm_baseline import VanillaLSTM
from src.models.social_stgcnn import SocialSTGCNN, bivariate_loss
from src.train_lstm import evaluate as eval_lstm
from src.train_lstm import l2_loss
from src.train_stgcnn import evaluate as eval_stgcnn
from src.train_stgcnn import make_loader, prepare_inputs
from src.data_loader import data_loader


def objective_lstm(trial: optuna.Trial, args) -> float:
    device = args.device
    torch.manual_seed(args.seed)

    # --- prostor pretrage ---
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    embedding_dim = trial.suggest_categorical("embedding_dim", [16, 32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 1, 2)
    dropout = trial.suggest_float("dropout", 0.0, 0.3)

    data_dir = os.path.join(args.data_root, args.scene)
    _, train_loader = data_loader(data_dir, "train", batch_size=args.batch_size, shuffle=True)
    _, val_loader = data_loader(data_dir, "val", batch_size=args.batch_size, shuffle=False)

    model = VanillaLSTM(
        embedding_dim=embedding_dim, hidden_dim=hidden_dim,
        num_layers=num_layers, dropout=dropout,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for obs, _, obs_rel, pred_rel_gt, _ in train_loader:
            obs_rel, pred_rel_gt = obs_rel.to(device), pred_rel_gt.to(device)
            opt.zero_grad()
            l2_loss(model(obs_rel), pred_rel_gt).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        val_ade, _ = eval_lstm(model, val_loader, device)
        best = min(best, val_ade)

        # pruning: prekid trial-a koji ocigledno zaostaje
        trial.report(val_ade, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best


def objective_stgcnn(trial: optuna.Trial, args) -> float:
    device = args.device
    torch.manual_seed(args.seed)

    # --- prostor pretrage ---
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    n_stgcnn = trial.suggest_int("n_stgcnn", 1, 3)          # broj graph konvolucionih blokova
    n_txpcnn = trial.suggest_int("n_txpcnn", 3, 7)          # dubina vremenskog dekodera
    kernel_size = trial.suggest_categorical("kernel_size", [3, 5])
    dropout = trial.suggest_float("dropout", 0.0, 0.3)

    data_dir = os.path.join(args.data_root, args.scene)
    _, train_loader = make_loader(data_dir, "train", True, args.batch_size)
    _, val_loader = make_loader(data_dir, "val", False, args.batch_size)

    model = SocialSTGCNN(
        n_stgcnn=n_stgcnn, n_txpcnn=n_txpcnn,
        kernel_size=kernel_size, dropout=dropout,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for obs, _, obs_rel, pred_rel_gt, sse in train_loader:
            obs, obs_rel, pred_rel_gt = obs.to(device), obs_rel.to(device), pred_rel_gt.to(device)
            v, a = prepare_inputs(obs, obs_rel, sse)
            out = model(v, a).squeeze(0).permute(1, 2, 0)
            opt.zero_grad()
            bivariate_loss(out, pred_rel_gt).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if epoch % args.val_every == 0 or epoch == args.epochs:
            _, _, val_ade, _ = eval_stgcnn(model, val_loader, device, n_samples=args.val_samples)
            best = min(best, val_ade)
            trial.report(val_ade, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="stgcnn", choices=["lstm", "stgcnn"])
    p.add_argument("--scene", default="zara1")
    p.add_argument("--data_root", default="data")
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--trials", type=int, default=15)
    p.add_argument("--epochs", type=int, default=40, help="epoha po trial-u (kraci trening nego finalni)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--val_every", type=int, default=5)
    p.add_argument("--val_samples", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2),
        study_name=f"{args.model}_{args.scene}",
    )

    objective = objective_lstm if args.model == "lstm" else objective_stgcnn

    def callback(study, trial):
        state = trial.state.name
        val = f"{trial.value:.4f}" if trial.value is not None else "pruned"
        print(f"  trial {trial.number:2d}  {state:<9}  val_ADE={val}  {trial.params}", flush=True)

    print(f"Optuna pretraga: model={args.model} scena={args.scene} trials={args.trials}\n")
    study.optimize(lambda t: objective(t, args), n_trials=args.trials, callbacks=[callback])

    print("\n" + "=" * 60)
    print(f"Najbolji val_ADE: {study.best_value:.4f} m")
    print("Najbolji hiperparametri:")
    for k, v in study.best_params.items():
        print(f"  {k:16s} = {v}")
    print("=" * 60)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"optuna_{args.model}_{args.scene}.json")
    with open(path, "w") as f:
        json.dump(
            {
                "model": args.model,
                "scene": args.scene,
                "best_value": study.best_value,
                "best_params": study.best_params,
                "n_trials": len(study.trials),
                "trials": [
                    {"number": t.number, "state": t.state.name, "value": t.value, "params": t.params}
                    for t in study.trials
                ],
            },
            f,
            indent=2,
        )
    print(f"\nRezultati sacuvani u {path}")
    print(f"Finalni trening sa najboljim parametrima npr.:")
    flags = " ".join(f"--{k} {v}" for k, v in study.best_params.items())
    print(f"  python -m src.train_{args.model} --scene {args.scene} --epochs 250 {flags}")


if __name__ == "__main__":
    main()
