#!/usr/bin/env bash
# Downloads the standardised ETH-UCY dataset into data/.
#
# Uses the split shipped with the Social-STGCNN repository, which is identical to the
# one from the Social-GAN paper (Gupta et al., CVPR 2018) -- the reference format in
# this field, which keeps our results directly comparable to published figures.
#
# Layout: data/<scene>/{train,val,test}/*.txt, format "frame_id ped_id x y".
# The split is leave-one-scene-out: e.g. data/eth/train holds the other four scenes,
# while data/eth/test holds ETH only.

set -euo pipefail

REPO="https://github.com/abduallahmohamed/Social-STGCNN.git"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data"

if [ -d "$DEST/eth/train" ]; then
    echo "Data already present in $DEST -- skipping download."
    exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Downloading ETH-UCY data..."
git clone --depth 1 --quiet "$REPO" "$TMP/stgcnn"

mkdir -p "$DEST"
for scene in eth hotel univ zara1 zara2; do
    cp -r "$TMP/stgcnn/datasets/$scene" "$DEST/"
done

echo "Done. Contents of $DEST:"
for scene in eth hotel univ zara1 zara2; do
    n=$(find "$DEST/$scene" -name '*.txt' | wc -l)
    echo "  $scene: $n files"
done
