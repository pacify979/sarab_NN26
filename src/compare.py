"""
Generise finalnu uporednu tabelu (Markdown) spremnu za ubacivanje u izvestaj.

    python -m src.compare

Cita outputs/lstm_*_log.json i outputs/stgcnn_*_log.json i pravi
outputs/rezultati.md sa poredjenjem oba modela po scenama, uz referentne
vrednosti iz literature radi kontrole ispravnosti.
"""

import argparse
import json
import os

SCENES = ["eth", "hotel", "univ", "zara1", "zara2"]

# Referentne vrednosti iz radova (ADE/FDE u metrima, protokol 8/12).
# Sluze kao "sanity check": ako su nasi brojevi u istom rangu, pipeline je ispravan.
LITERATURE = {
    "LSTM (Social-GAN rad, deterministicki)": {
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
    lines.append("# Rezultati\n")
    lines.append("Protokol: 8 osmotrenih frejmova (3.2 s) -> 12 predvidjenih (4.8 s), "
                 "leave-one-scene-out. Sve vrednosti u metrima, manje je bolje.\n")

    lines.append("## Poredjenje 1: deterministicki modeli (L2 gubitak, jedna predvidjena putanja)\n")
    lines.append("| Scena | LSTM ADE | LSTM FDE | ST-GCNN ADE | ST-GCNN FDE |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['scene'].upper()} | {fmt(r['lstm_ade'])} | {fmt(r['lstm_fde'])} "
            f"| {fmt(r['stg_ade_det'])} | {fmt(r['stg_fde_det'])} |"
        )
    lines.append(
        f"| **PROSEK** | **{fmt(avg('lstm_ade'))}** | **{fmt(avg('lstm_fde'))}** "
        f"| **{fmt(avg('stg_ade_det'))}** | **{fmt(avg('stg_fde_det'))}** |"
    )
    a_l, a_s = avg("lstm_ade"), avg("stg_ade_det")
    if a_l and a_s:
        lines.append(
            f"\n> Napomena: ova kolona je samo uslovno posteno poredjenje. ST-GCNN je treniran "
            f"NLL gubitkom (uci raspodelu), pa njegova srednja vrednost nije optimizovana za ADE. "
            f"Razlika od {100*(a_l-a_s)/a_l:+.1f}% meri i razliku u funkciji gubitka, ne samo "
            f"doprinos grafa.\n"
        )

    lines.append("\n## Poredjenje 2: probabilisticki modeli (isti NLL gubitak, best-of-20)\n")
    lines.append("Ovo je **kljucno poredjenje** rada: oba modela imaju identicnu bivarijantnu "
                 "Gausovu glavu, isti gubitak i isti protokol evaluacije. Jedina razlika je da li "
                 "model vidi druge agente. Zato razlika meri iskljucivo doprinos socijalnog "
                 "modelovanja.\n")
    lines.append("| Scena | LSTM-prob ADE | LSTM-prob FDE | ST-GCNN ADE | ST-GCNN FDE |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['scene'].upper()} | {fmt(r['prob_ade_20'])} | {fmt(r['prob_fde_20'])} "
            f"| {fmt(r['stg_ade_20'])} | {fmt(r['stg_fde_20'])} |"
        )
    lines.append(
        f"| **PROSEK** | **{fmt(avg('prob_ade_20'))}** | **{fmt(avg('prob_fde_20'))}** "
        f"| **{fmt(avg('stg_ade_20'))}** | **{fmt(avg('stg_fde_20'))}** |"
    )
    a_p, a_g = avg("prob_ade_20"), avg("stg_ade_20")
    if a_p and a_g:
        lines.append(
            f"\n**Razlika ST-GCNN vs. LSTM-prob:** ADE {100*(a_p-a_g)/a_p:+.1f}% "
            f"(pozitivno = ST-GCNN bolji)\n"
        )

    lines.append("\n## Referentne vrednosti iz literature\n")
    lines.append("Nasi rezultati treba da budu u istom rangu -- to potvrdjuje da su "
                 "protokol i evaluacija ispravno implementirani.\n")
    for name, vals in LITERATURE.items():
        lines.append(f"\n**{name}**\n")
        lines.append("| Scena | " + " | ".join(s.upper() for s in SCENES) + " | Prosek |")
        lines.append("|---|" + "---|" * (len(SCENES) + 1))
        ade = [vals[s][0] for s in SCENES]
        fde = [vals[s][1] for s in SCENES]
        lines.append("| ADE | " + " | ".join(f"{v:.2f}" for v in ade) + f" | {sum(ade)/len(ade):.2f} |")
        lines.append("| FDE | " + " | ".join(f"{v:.2f}" for v in fde) + f" | {sum(fde)/len(fde):.2f} |")

    lines.append("\n## Metodoloske napomene\n")
    lines.append(
        "- ADE = srednja **euklidska** udaljenost kroz svih 12 koraka [m]; FDE = udaljenost samo "
        "u poslednjem koraku [m]. ADE **nije** srednja kvadratna greska.\n"
        "- **Best-of-20** je standardni protokol iz literature za probabilisticke modele: "
        "uzorkuje se 20 trajektorija i uzima se najbolja (minimum po agentu). Brojke iz tog "
        "protokola nisu uporedive sa deterministickim, pa su tabele razdvojene.\n"
        "- Sva tri modela treniraju se i biraju na osnovu **validacionog** skupa; test skup se "
        "koristi iskljucivo za finalnu evaluaciju.\n"
        "- ETH je konzistentno najteza scena kod svih modela -- to je poznato i u literaturi "
        "(drugaciji ugao kamere i skala scene), nije posledica implementacije.\n"
    )

    out = os.path.join(args.out_dir, "rezultati.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[sacuvano u {out}]")


if __name__ == "__main__":
    main()
