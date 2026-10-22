"""Build the register templates Stage 1 needs for a new data domain.

Rewrites the first --n_texts texts with each register prompt (24 in data/register_prompts.json),
then asks the LLM to abstract each register's first sentences into one template.
Use your suspect texts (or other in-domain texts) as --texts.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sdr.templates import build_reference_rewrites, build_templates  # noqa: E402
from sdr.utils import load_json, require_api_key, save_json, set_seed_all  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--texts", required=True, help="JSON list of in-domain texts (at least --n_texts)")
    p.add_argument("--register_prompts", default=os.path.join(ROOT, "data", "register_prompts.json"))
    p.add_argument("--gpt_model", default="gpt-4o-mini")
    p.add_argument("--n_texts", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/templates/templates.json")
    p.add_argument("--rewrites_out", default=None, help="where to keep the reference rewrites (default: next to --out)")
    return p.parse_args()


def main():
    args = parse_args()
    texts = load_json(args.texts)
    if len(texts) < args.n_texts:
        raise SystemExit(f"--texts has {len(texts)} texts; need at least --n_texts={args.n_texts}")
    require_api_key()
    set_seed_all(args.seed)
    rewrites = build_reference_rewrites(texts, load_json(args.register_prompts), args.gpt_model, args.n_texts)
    rewrites_out = args.rewrites_out or os.path.splitext(args.out)[0] + "_reference_rewrites.json"
    save_json(rewrites, rewrites_out)
    save_json(build_templates(rewrites, args.n_texts), args.out)
    print(f"Templates: {args.out}\nReference rewrites: {rewrites_out}")


if __name__ == "__main__":
    main()
