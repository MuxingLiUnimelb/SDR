"""Demo: run SDR on released Pythia-6.9b models (MIMIR-wiki laundered with the paper's inside prompts).

By default the five models laundered with inside prompts 2-6 of the paper (spoken style, interview, interactive
discussion, storytelling, news report). Uses the shipped data split and the LoRA adapters from the GitHub Release
(`bash scripts/download_release.sh inside`, plus `caches` for --replay). Nothing is regenerated or trained.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sdr.paper import detect_command, run_logged, select  # noqa: E402
from sdr.pipeline import avg_auc  # noqa: E402
from sdr.utils import load_json, require_api_key  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids", type=int, nargs="+", default=[2, 3, 4, 5, 6], choices=range(1, 24), metavar="ID",
                   help="inside experiments (paper appendix IDs 1-23); default 2 3 4 5 6")
    p.add_argument("--replay", action="store_true",
                   help="replay the LLM calls recorded for the paper instead of calling the API (exact, no API key)")
    p.add_argument("--rounds", type=int, default=10, help="Stage-2 rounds (with --replay: replays the first rounds)")
    p.add_argument("--adapter_dir", default=os.path.join(ROOT, "adapters"))
    p.add_argument("--cache_dir", default=os.path.join(ROOT, "caches"))
    p.add_argument("--model_path", default=None, help="local Pythia-6.9b weights (optional)")
    p.add_argument("--ref_model_path", default=None, help="local LiteLlama-460M-1T weights (optional)")
    p.add_argument("--deterministic_baseline", action="store_true",
                   help="turn the LoRA dropout off for the baseline MIA (not with --replay; see README)")
    p.add_argument("--out_dir", default=os.path.join(ROOT, "results", "demo"))
    return p.parse_args()


def main():
    args = parse_args()
    if args.replay and args.deterministic_baseline:
        raise SystemExit("--deterministic_baseline changes the random stream, so the recorded calls no longer match; "
                         "drop it or drop --replay")
    if not 1 <= args.rounds <= 10 and args.replay:
        raise SystemExit("--replay supports 1-10 rounds (the paper's runs had 10)")
    for k in args.ids:
        needed = [os.path.join(args.adapter_dir, "inside", str(k), "adapter_config.json")]
        if args.replay:
            needed.append(os.path.join(args.cache_dir, "inside", f"{k}.jsonl.gz"))
        for path in needed:
            if not os.path.exists(path):
                raise SystemExit(f"{path} not found. Run `bash scripts/download_release.sh inside"
                                 f"{' caches' if args.replay else ''}` first.")
    if not args.replay:
        require_api_key()
    secret = load_json(os.path.join(ROOT, "data", "secret_prompts.json"))["inside"]

    rows = []
    for k in args.ids:
        name = f"inside-{k}"
        print(f"\n===== {name}: {secret[str(k)]} =====", flush=True)
        cmd = detect_command(name, args.adapter_dir, args.out_dir, cache_dir=args.cache_dir if args.replay else None,
                             rounds=args.rounds, model_path=args.model_path, ref_model_path=args.ref_model_path,
                             extra=["--deterministic_baseline"] if args.deterministic_baseline else [])
        log_path = os.path.join(args.out_dir, f"{name}.log")
        if run_logged(cmd, log_path) != 0:
            raise SystemExit(f"{name} failed; see {log_path}")
        result = json.load(open(os.path.join(args.out_dir, f"{name}.json")))
        prompt, sdr = select(result, open(log_path, errors="replace").read())
        rows.append((k, avg_auc(result["original"]), avg_auc(sdr), prompt))

    print("\n id | baseline | SDR   | secret prompt / recovered prompt")
    for k, baseline, best, prompt in rows:
        print(f" {k:2d} |  {baseline:.3f}   | {best:.3f} | secret:    {secret[str(k)][:90]}")
        print(f"    |          |       | recovered: {(prompt or '-')[:90]}")
    print(f"\nScores are the mean AUC of the five MIAs. Per-experiment results and logs: {args.out_dir}")


if __name__ == "__main__":
    main()
