#!/usr/bin/env bash
# Download the LoRA adapters used by scripts/demo.py:
# Pythia-6.9b fine-tuned on MIMIR-wiki texts laundered with inside prompts 1-5.
# Usage: bash scripts/download_adapters.sh [DEST]   (default DEST=adapters)
set -euo pipefail
URL="${SDR_ADAPTER_URL:-https://github.com/OWNER/SDR/releases/download/v1.0/adapters_pythia-6.9b_wiki_prompt1-5.zip}"
DEST="${1:-adapters}"
if [[ "$URL" == *"/OWNER/"* ]]; then
  echo "The release URL is not configured yet; set SDR_ADAPTER_URL to the adapters zip URL." >&2
  exit 1
fi
mkdir -p "$DEST"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -L --fail -o "$tmp/adapters.zip" "$URL"
unzip -q -o "$tmp/adapters.zip" -d "$DEST"
for k in 1 2 3 4 5; do
  test -f "$DEST/prompt$k/adapter_model.safetensors" || { echo "missing $DEST/prompt$k" >&2; exit 1; }
done
echo "Adapters ready in $DEST/"
