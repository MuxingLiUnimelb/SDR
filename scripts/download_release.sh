#!/usr/bin/env bash
# Download the paper's LoRA adapters and recorded LLM calls from the GitHub Release into the repository.
# Usage: bash scripts/download_release.sh [inside] [outside] [caches]   (default: all three)
#   inside   adapters/inside/1..23    Pythia-6.9b adapters, inside experiments (the demo uses 2-6)   ~580 MB
#   outside  adapters/outside/1..10   Pythia-6.9b adapters, outside experiments                      ~250 MB
#   caches   caches/{inside,outside}/<id>.jsonl.gz   recorded LLM calls, for exact replay           ~60 MB
set -euo pipefail
BASE="${SDR_RELEASE_URL:-https://github.com/OWNER/REPO/releases/download/v1.0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ "$BASE" == *"/OWNER/"* ]]; then
  echo "The release URL is not configured yet; set SDR_RELEASE_URL to the release download URL." >&2
  exit 1
fi
parts=("$@")
[ ${#parts[@]} -eq 0 ] && parts=(inside outside caches)
if command -v sha256sum >/dev/null; then check=(sha256sum -c -); else check=(shasum -a 256 -c -); fi
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -L --fail -o "$tmp/SHA256SUMS" "$BASE/SHA256SUMS"
for part in "${parts[@]}"; do
  case "$part" in
    inside)  file=sdr_adapters_inside.zip;  expect=(adapters/inside/{1..23}/adapter_model.safetensors) ;;
    outside) file=sdr_adapters_outside.zip; expect=(adapters/outside/{1..10}/adapter_model.safetensors) ;;
    caches)  file=sdr_llm_caches.zip;       expect=(caches/inside/{1..23}.jsonl.gz caches/outside/{1..10}.jsonl.gz) ;;
    *) echo "unknown part '$part' (choose from: inside outside caches)" >&2; exit 1 ;;
  esac
  curl -L --fail -o "$tmp/$file" "$BASE/$file"
  (cd "$tmp" && grep " $file\$" SHA256SUMS | "${check[@]}")
  unzip -q -o "$tmp/$file" -d "$ROOT"
  for f in "${expect[@]}"; do
    test -f "$ROOT/$f" || { echo "missing $f after unpacking $file" >&2; exit 1; }
  done
  echo "$part ready"
done
