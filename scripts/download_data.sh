#!/usr/bin/env bash
# Preuzima standardizovani ETH-UCY skup podataka u data/.
#
# Koristi se split iz Social-STGCNN repozitorijuma, koji je identican onom iz
# Social-GAN rada (Gupta et al., CVPR 2018) -- to je referentni format u ovoj
# oblasti, pa su nasi rezultati direktno uporedivi sa objavljenim brojkama.
#
# Struktura: data/<scena>/{train,val,test}/*.txt, format "frame_id ped_id x y".
# Split je leave-one-scene-out: npr. data/eth/train sadrzi preostale 4 scene,
# a data/eth/test iskljucivo ETH.

set -euo pipefail

REPO="https://github.com/abduallahmohamed/Social-STGCNN.git"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data"

if [ -d "$DEST/eth/train" ]; then
    echo "Podaci vec postoje u $DEST -- preskacem preuzimanje."
    exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Preuzimanje ETH-UCY podataka..."
git clone --depth 1 --quiet "$REPO" "$TMP/stgcnn"

mkdir -p "$DEST"
for scene in eth hotel univ zara1 zara2; do
    cp -r "$TMP/stgcnn/datasets/$scene" "$DEST/"
done

echo "Gotovo. Sadrzaj $DEST:"
for scene in eth hotel univ zara1 zara2; do
    n=$(find "$DEST/$scene" -name '*.txt' | wc -l)
    echo "  $scene: $n fajlova"
done
