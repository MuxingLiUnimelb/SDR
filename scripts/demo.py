"""Demo: run SDR on the five released Pythia-6.9b models (MIMIR-wiki laundered with inside prompts 1-5).

Uses the shipped data split and the LoRA adapters from the GitHub Release
(download them first with `bash scripts/download_adapters.sh`). No data is regenerated and nothing is trained.
"""
import argparse
import gc
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sdr.pipeline import check_inputs, run_detection, summarize  # noqa: E402
from sdr.utils import load_json, load_model, require_api_key, save_json, set_seed_all  # noqa: E402

DATA = os.path.join(ROOT, "data")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt_ids", type=int, nargs="+", default=[1, 2, 3, 4, 5], choices=[1, 2, 3, 4, 5])
    p.add_argument("--adapter_dir", default=os.path.join(ROOT, "adapters"))
    p.add_argument("--model", default="EleutherAI/pythia-6.9b")
    p.add_argument("--model_path", default=None, help="local Pythia-6.9b weights (optional)")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--l", type=int, default=5)
    p.add_argument("--m", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default=os.path.join(ROOT, "results", "demo"))
    return p.parse_args()


def main():
    args = parse_args()
    for k in args.prompt_ids:
        if not os.path.isfile(os.path.join(args.adapter_dir, f"prompt{k}", "adapter_config.json")):
            raise SystemExit(f"Adapter for prompt {k} not found in {args.adapter_dir}/prompt{k}. "
                             f"Run `bash scripts/download_adapters.sh` first.")
    suspect = load_json(os.path.join(DATA, "mimir_wiki", "members.json"))
    reference = load_json(os.path.join(DATA, "mimir_wiki", "nonmembers.json"))
    prefix = load_json(os.path.join(DATA, "mimir_wiki", "prefix.json"))
    secret = load_json(os.path.join(DATA, "secret_prompts.json"))
    check_inputs(suspect, reference, prefix)
    require_api_key()

    rows = []
    for k in args.prompt_ids:
        print(f"\n===== prompt {k}: {secret[str(k)]} =====")
        set_seed_all(args.seed)
        model, tokenizer = load_model(args.model, args.model_path, os.path.join(args.adapter_dir, f"prompt{k}"))
        result = run_detection(model, tokenizer, suspect, reference, prefix,
                               os.path.join(DATA, "templates", "mimir_wiki.json"),
                               os.path.join(DATA, "register_prompts.json"),
                               out_dir=os.path.join(args.out_dir, f"prompt{k}"), model_name=args.model,
                               rounds=args.rounds, l=args.l, m=args.m, prefix_size=len(prefix))
        save_json(result, os.path.join(args.out_dir, f"prompt{k}.json"))
        rows.append((k,) + summarize(result))
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print("\n id | baseline | SDR best | secret prompt / recovered prompt")
    for k, baseline, best, best_prompt in rows:
        best_s = f"{best:.3f}" if best is not None else "  -  "
        print(f" {k:2d} |  {baseline:.3f}   |  {best_s}   | secret:    {secret[str(k)][:90]}")
        print(f"    |          |          | recovered: {(best_prompt or '-')[:90]}")
    print(f"\nPer-prompt results saved in {args.out_dir}")


if __name__ == "__main__":
    main()
