# Rezultati

Protokol: 8 osmotrenih frejmova (3.2 s) -> 12 predvidjenih (4.8 s), leave-one-scene-out. Sve vrednosti u metrima, manje je bolje.

## Poredjenje 1: deterministicki modeli (L2 gubitak, jedna predvidjena putanja)

| Scena | LSTM ADE | LSTM FDE | ST-GCNN ADE | ST-GCNN FDE |
|---|---|---|---|---|
| ETH | 1.007 | 1.996 | 1.043 | 2.113 |
| HOTEL | 0.457 | 0.936 | 0.436 | 0.880 |
| UNIV | 0.577 | 1.278 | 0.568 | 1.202 |
| ZARA1 | 0.402 | 0.886 | 0.419 | 0.904 |
| ZARA2 | 0.316 | 0.699 | 0.350 | 0.761 |
| **PROSEK** | **0.552** | **1.159** | **0.563** | **1.172** |

> Napomena: ova kolona je samo uslovno posteno poredjenje. ST-GCNN je treniran NLL gubitkom (uci raspodelu), pa njegova srednja vrednost nije optimizovana za ADE. Razlika od -2.1% meri i razliku u funkciji gubitka, ne samo doprinos grafa.


## Poredjenje 2: probabilisticki modeli (isti NLL gubitak, best-of-20)

Ovo je **kljucno poredjenje** rada: oba modela imaju identicnu bivarijantnu Gausovu glavu, isti gubitak i isti protokol evaluacije. Jedina razlika je da li model vidi druge agente. Zato razlika meri iskljucivo doprinos socijalnog modelovanja.

| Scena | LSTM-prob ADE | LSTM-prob FDE | ST-GCNN ADE | ST-GCNN FDE |
|---|---|---|---|---|
| ETH | 0.781 | 1.422 | 0.819 | 1.582 |
| HOTEL | 0.368 | 0.703 | 0.311 | 0.561 |
| UNIV | 0.392 | 0.756 | 0.412 | 0.803 |
| ZARA1 | 0.281 | 0.512 | 0.283 | 0.498 |
| ZARA2 | 0.225 | 0.427 | 0.251 | 0.474 |
| **PROSEK** | **0.410** | **0.764** | **0.415** | **0.784** |

**Razlika ST-GCNN vs. LSTM-prob:** ADE -1.4% (pozitivno = ST-GCNN bolji)


## Referentne vrednosti iz literature

Nasi rezultati treba da budu u istom rangu -- to potvrdjuje da su protokol i evaluacija ispravno implementirani.


**LSTM (Social-GAN rad, deterministicki)**

| Scena | ETH | HOTEL | UNIV | ZARA1 | ZARA2 | Prosek |
|---|---|---|---|---|---|---|
| ADE | 1.09 | 0.86 | 0.61 | 0.41 | 0.52 | 0.70 |
| FDE | 2.94 | 1.91 | 1.31 | 0.88 | 1.11 | 1.63 |

**Social-STGCNN (CVPR 2020, best-of-20)**

| Scena | ETH | HOTEL | UNIV | ZARA1 | ZARA2 | Prosek |
|---|---|---|---|---|---|---|
| ADE | 0.64 | 0.49 | 0.44 | 0.34 | 0.30 | 0.44 |
| FDE | 1.11 | 0.85 | 0.79 | 0.53 | 0.48 | 0.75 |

## Metodoloske napomene

- ADE = srednja **euklidska** udaljenost kroz svih 12 koraka [m]; FDE = udaljenost samo u poslednjem koraku [m]. ADE **nije** srednja kvadratna greska.
- **Best-of-20** je standardni protokol iz literature za probabilisticke modele: uzorkuje se 20 trajektorija i uzima se najbolja (minimum po agentu). Brojke iz tog protokola nisu uporedive sa deterministickim, pa su tabele razdvojene.
- Sva tri modela treniraju se i biraju na osnovu **validacionog** skupa; test skup se koristi iskljucivo za finalnu evaluaciju.
- ETH je konzistentno najteza scena kod svih modela -- to je poznato i u literaturi (drugaciji ugao kamere i skala scene), nije posledica implementacije.

