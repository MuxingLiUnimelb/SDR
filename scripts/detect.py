"""Run SDR against a suspect model.

Given texts you suspect were rewritten and used for fine-tuning (--suspect), comparable texts you know
were not (--reference) and a few extra non-member texts for the ReCaLL prefix (--prefix), SDR recovers
a rewriting prompt and reports membership-inference AUCs on the re-rewritten texts, next to the
baseline MIA on the original texts ("original").
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sdr.pipeline import check_files, check_inputs, run_detection, summarize, work_dir_for  # noqa: E402
from sdr.utils import load_json, load_model, load_ref_model, require_api_key, save_json, set_seed_all  # noqa: E402
from sdr.gpt_cache import enable as enable_gpt_cache  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suspect", required=True, help="JSON list of suspect texts (>= 100)")
    p.add_argument("--reference", required=True, help="JSON list of known non-member texts (>= 100)")
    p.add_argument("--prefix", required=True, help="JSON list of non-member texts for the ReCaLL prefix (7 in the paper)")
    p.add_argument("--model", default="EleutherAI/pythia-6.9b", help="Hugging Face id of the suspect model's base")
    p.add_argument("--model_path", default=None, help="local weights for --model (optional)")
    p.add_argument("--adapter", default=None, help="LoRA adapter of the suspect model (optional)")
    p.add_argument("--ref_model", default="ahxt/LiteLlama-460M-1T", help="reference model for the `ref` metric")
    p.add_argument("--ref_model_path", default=None, help="local weights for --ref_model (optional)")
    p.add_argument("--deterministic_baseline", action="store_true",
                   help="turn the LoRA dropout off for the baseline MIA (the paper kept it on until Stage 1; see README)")
    p.add_argument("--templates", default=os.path.join(ROOT, "data", "templates", "mimir_wiki.json"),
                   help="register templates (scripts/build_templates.py)")
    p.add_argument("--register_prompts", default=os.path.join(ROOT, "data", "register_prompts.json"))
    p.add_argument("--rounds", type=int, default=10, help="Stage-2 refinement rounds (K)")
    p.add_argument("--l", type=int, default=5, help="texts per Stage-2 round")
    p.add_argument("--m", type=int, default=5, help="texts used for Stage-1 register identification")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/detect/result.json")
    p.add_argument("--gpt_cache", default=None,
                   help="record/replay all LLM calls in this JSONL (.jsonl.gz to replay only); see README")
    p.add_argument("--gpt_cache_strict", action="store_true",
                   help="replay only: fail on any LLM call missing from --gpt_cache (no API key needed)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.gpt_cache_strict and not args.gpt_cache:
        raise SystemExit("--gpt_cache_strict needs --gpt_cache")
    suspect, reference, prefix = load_json(args.suspect), load_json(args.reference), load_json(args.prefix)
    check_inputs(suspect, reference, prefix)
    check_files(args.templates, args.register_prompts, args.adapter)
    cache = enable_gpt_cache(args.gpt_cache, args.gpt_cache_strict) if args.gpt_cache else None
    if not args.gpt_cache_strict:
        require_api_key()
    set_seed_all(args.seed)
    model, tokenizer = load_model(args.model, args.model_path, args.adapter, paper_dropout=not args.deterministic_baseline)
    ref_model = load_ref_model(args.ref_model, args.ref_model_path)
    result = run_detection(model, tokenizer, ref_model, suspect, reference, prefix, args.templates, args.register_prompts,
                           out_dir=work_dir_for(args.out), model_name=args.model, ref_model_name=args.ref_model,
                           rounds=args.rounds, l=args.l, m=args.m, prefix_size=len(prefix))
    save_json(result, args.out)
    baseline, best, best_prompt = summarize(result)
    print(f"Baseline MIA on original texts (mean AUC): {baseline:.3f}")
    if best is None:
        print("SDR: no Stage-2 round passed the 0.65 gate; see the JSON for details.")
    else:
        print(f"SDR best (mean AUC): {best:.3f}\nRecovered prompt: {best_prompt}")
    if cache is not None:
        print(f"LLM cache {args.gpt_cache}: {cache.hits} calls replayed, {cache.misses} recorded")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
