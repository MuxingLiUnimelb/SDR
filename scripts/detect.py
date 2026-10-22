"""Run SDR against a suspect model.

Given texts you suspect were rewritten and used for fine-tuning (--suspect), comparable texts you know
were not (--reference) and a few extra non-member texts for the ReCaLL prefix (--prefix), SDR recovers
a rewriting prompt and reports membership-inference AUCs on the re-rewritten texts, next to the
baseline MIA on the original texts ("original").
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdr.pipeline import check_inputs, run_detection, summarize  # noqa: E402
from sdr.utils import load_json, load_model, require_api_key, save_json, set_seed_all  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suspect", required=True, help="JSON list of suspect texts (>= 100)")
    p.add_argument("--reference", required=True, help="JSON list of known non-member texts (>= 100)")
    p.add_argument("--prefix", required=True, help="JSON list of non-member texts for the ReCaLL prefix (7 in the paper)")
    p.add_argument("--model", default="EleutherAI/pythia-6.9b", help="Hugging Face id of the suspect model's base")
    p.add_argument("--model_path", default=None, help="local weights for --model (optional)")
    p.add_argument("--adapter", default=None, help="LoRA adapter of the suspect model (optional)")
    p.add_argument("--templates", default="data/templates/mimir_wiki.json", help="register templates (scripts/build_templates.py)")
    p.add_argument("--register_prompts", default="data/register_prompts.json")
    p.add_argument("--rounds", type=int, default=10, help="Stage-2 refinement rounds (K)")
    p.add_argument("--l", type=int, default=5, help="texts per Stage-2 round")
    p.add_argument("--m", type=int, default=5, help="texts used for Stage-1 register identification")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/detect/result.json")
    return p.parse_args()


def main():
    args = parse_args()
    suspect, reference, prefix = load_json(args.suspect), load_json(args.reference), load_json(args.prefix)
    check_inputs(suspect, reference, prefix)
    require_api_key()
    set_seed_all(args.seed)
    model, tokenizer = load_model(args.model, args.model_path, args.adapter)
    result = run_detection(model, tokenizer, suspect, reference, prefix, args.templates, args.register_prompts,
                           out_dir=os.path.dirname(os.path.abspath(args.out)), model_name=args.model,
                           rounds=args.rounds, l=args.l, m=args.m, prefix_size=len(prefix))
    save_json(result, args.out)
    baseline, best, best_prompt = summarize(result)
    print(f"Baseline MIA on original texts (mean AUC): {baseline:.3f}")
    if best is None:
        print("SDR: no Stage-2 round passed the 0.65 gate; see the JSON for details.")
    else:
        print(f"SDR best (mean AUC): {best:.3f}\nRecovered prompt: {best_prompt}")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
