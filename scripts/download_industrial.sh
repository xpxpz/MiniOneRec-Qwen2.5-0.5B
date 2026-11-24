#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
origin=https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2
download() {
    local source="$1" destination="data/raw/${1##*/}"
    if [[ -f "$destination" ]] && gzip -t "$destination"; then
        return
    fi
    curl -fL --retry 3 -C - "$origin/$source" -o "$destination.part"
    gzip -t "$destination.part"
    mv "$destination.part" "$destination"
}
download categoryFilesSmall/Industrial_and_Scientific_5.json.gz
download metaFiles2/meta_Industrial_and_Scientific.json.gz
