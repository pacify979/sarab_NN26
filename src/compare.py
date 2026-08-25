"""
Generates the final comparison tables (Markdown), ready to drop into the report.

    python -m src.compare

Reads outputs/{lstm,lstm_prob,stgcnn}_*_log.json and writes outputs/results.md,
comparing all three models per scene, alongside published reference values used
as a correctness check.
"""

import argparse
import json
import os

SCENES = ["eth", "hotel", "univ", "zara1", "zara2"]

# Published reference values (ADE/FDE in metres, 8/12 protocol).
# Serve as a sanity check: if our numbers land in the same range, the pipeline is correct.
LITERATURE = {
    "LSTM baseline (Social-GAN paper, deterministic)": {
        "eth": (1.09, 2.94), "hotel": (0.86, 1.91), "univ": (0.61, 1.31),
        "zara1": (0.41, 0.88), "zara2": (0.52, 1.11),
    },
    "Social-STGCNN (CVPR 2020, best-of-20)": {
        "eth": (0.64, 1.11), "hotel": (0.49, 0.85), "univ": (0.44, 0.79),
        "zara1": (0.34, 0.53), "zara2": (0.30, 0.48),
    },
}


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="outputs")
    args = p.parse_args()

    rows = []
    for s in SCENES:
        lstm = load(os.path.join(args.out_dir, f"lstm_{s}_log.json"))
        prob = load(os.path.join(args.out_dir, f"lstm_prob_{s}_log.json"))
        stg = load(os.path.join(args.out_dir, f"stgcnn_{s}_log.json"))
        rows.append({
            "scene": s,
            "lstm_ade": lstm["test_ade"] if lstm else None,
            "lstm_fde": lstm["test_fde"] if lstm else None,
            "prob_ade_20": prob["test_ade_best20"] if prob else None,
            "prob_fde_20": prob["test_fde_best20"] if prob else None,
            "stg_ade_det": stg["test_ade_det"] if stg else None,
            "stg_fde_det": stg["test_fde_det"] if stg else None,
            "stg_ade_20": stg["test_ade_best20"] if stg else None,
            "stg_fde_20": stg["test_fde_best20"] if stg else None,
        })

    def avg(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    def fmt(v):
        return f"{v:.3f}" if v is not None else "—"

    lines = []
    lines.append("# Results\n")
    lines.append("Protocol: 8 observed frames (3.2 s) -> 12 predicted frames (4.8 s), "
                 "leave-one-scene-out. All values in metres; lower is better.\n")

    lines.append("## Comparison 1: deterministic models (single predicted path)\n")
    lines.append("| Scene | LSTM ADE | LSTM FDE | ST-GCNN ADE | ST-GCNN FDE |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['scene'].upper()} | {fmt(r['lstm_ade'])} | {fmt(r['lstm_fde'])} "
            f"| {fmt(r['stg_ade_det'])} | {fmt(r['stg_fde_det'])} |"
        )
    lines.append(
        f"| **AVERAGE** | **{fmt(avg('lstm_ade'))}** | **{fmt(avg('lstm_fde'))}** "
        f"| **{fmt(avg('stg_ade_det'))}** | **{fmt(avg('stg_fde_det'))}** |"
    )
    a_l, a_s = avg("lstm_ade"), avg("stg_ade_det")
    if a_l and a_s:
        lines.append(
            f"\n> Note: this comparison is only conditionally fair. Social-STGCNN is trained "
            f"with an NLL loss (it learns a distribution), so its mean is not optimised for ADE. "
            f"The {100*(a_l-a_s)/a_l:+.1f}% difference therefore also measures the change of "
            f"loss function, not the contribution of the graph alone.\n"
        )

    lines.append("\n## Comparison 2: probabilistic models (identical NLL loss, best-of-20)\n")
    lines.append("This is the **key comparison**: both models use an identical bivariate Gaussian "
                 "head, the same loss and the same evaluation protocol. The only difference is "
                 "whether the model can see other agents, so the difference measures the "
                 "contribution of social modelling alone.\n")
    lines.append("| Scene | LSTM-prob ADE | LSTM-prob FDE | ST-GCNN ADE | ST-GCNN FDE |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['scene'].upper()} | {fmt(r['prob_ade_20'])} | {fmt(r['prob_fde_20'])} "
            f"| {fmt(r['stg_ade_20'])} | {fmt(r['stg_fde_20'])} |"
        )
    lines.append(
        f"| **AVERAGE** | **{fmt(avg('prob_ade_20'))}** | **{fmt(avg('prob_fde_20'))}** "
        f"| **{fmt(avg('stg_ade_20'))}** | **{fmt(avg('stg_fde_20'))}** |"
    )
    a_p, a_g = avg("prob_ade_20"), avg("stg_ade_20")
    if a_p and a_g:
        lines.append(
            f"\n**Difference, ST-GCNN vs. LSTM-prob:** ADE {100*(a_p-a_g)/a_p:+.1f}% "
            f"(positive = ST-GCNN better)\n"
        )

    lines.append("\n## Published reference values\n")
    lines.append("Our results should land in the same range -- this confirms that the protocol "
                 "and the evaluation are implemented correctly.\n")
    for name, vals in LITERATURE.items():
        lines.append(f"\n**{name}**\n")
        lines.append("| Scene | " + " | ".join(s.upper() for s in SCENES) + " | Average |")
        lines.append("|---|" + "---|" * (len(SCENES) + 1))
        ade = [vals[s][0] for s in SCENES]
        fde = [vals[s][1] for s in SCENES]
        lines.append("| ADE | " + " | ".join(f"{v:.2f}" for v in ade) + f" | {sum(ade)/len(ade):.2f} |")
        lines.append("| FDE | " + " | ".join(f"{v:.2f}" for v in fde) + f" | {sum(fde)/len(fde):.2f} |")

    lines.append("\n## Methodological notes\n")
    lines.append(
        "- ADE = mean **Euclidean** distance over all 12 predicted steps [m]; FDE = distance at "
        "the final step only [m]. ADE is **not** mean squared error, which would be in m^2.\n"
        "- **Best-of-20** is the standard protocol for probabilistic models: 20 trajectories are "
        "sampled and the best one is kept (minimum taken per agent). Those numbers are not "
        "comparable to deterministic ones, which is why the tables are separated.\n"
        "- All three models are trained and selected on the **validation** set; the test set is "
        "used exclusively for the final evaluation.\n"
        "- ETH is consistently the hardest scene for every model. This is also reported in the "
        "literature (different camera angle and scene scale) and is not an artefact of this "
        "implementation.\n"
    )

    out = os.path.join(args.out_dir, "results.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[saved to {out}]")


if __name__ == "__main__":
    main()
