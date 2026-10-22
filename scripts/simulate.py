"""Simulate data laundering: rewrite texts with a secret prompt, then LoRA-fine-tune a model on the rewrites.

Defaults match the released adapters (LoRA r=8, alpha=32, lr 4e-4, 60 epochs, batch size 2,
one optimizer step per epoch). Pass --laundered to reuse existing rewrites (e.g. data/laundered/prompt4/members.json)
and skip the API entirely.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdr.utils import load_json, load_model, require_api_key, save_json, set_seed_all  # noqa: E402

# LoRA target modules are defined per model in sdr/finetune.py::train_model.
SUPPORTED_MODELS = ["EleutherAI/pythia-6.9b", "meta-llama/Llama-2-7b-hf", "mistralai/Mistral-7B-v0.3", "tiiuae/falcon-7b"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--members", help="JSON list of original texts to launder (required unless --laundered)")
    p.add_argument("--secret_prompt", help="the launderer's rewriting prompt (required unless --laundered)")
    p.add_argument("--laundered", help="JSON list of already-rewritten texts; skips rewriting")
    p.add_argument("--model", default="EleutherAI/pythia-6.9b", choices=SUPPORTED_MODELS)
    p.add_argument("--model_path", default=None, help="local weights for --model (optional)")
    p.add_argument("--lr", type=float, default=0.0004)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lora_dim", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--exp_name", default="simulate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="runs/simulate")
    args = p.parse_args()
    if not args.laundered and not (args.members and args.secret_prompt):
        p.error("give either --laundered, or both --members and --secret_prompt")
    return args


def main():
    args = parse_args()
    if not args.laundered:
        require_api_key()
    set_seed_all(args.seed)

    if args.laundered:
        laundered = load_json(args.laundered)
    else:
        from sdr.pipeline import launder
        laundered = launder(load_json(args.members), args.secret_prompt)
    save_json(laundered, os.path.join(args.out_dir, "laundered.json"))

    from sdr.finetune import train_model
    model, tokenizer = load_model(args.model, args.model_path)
    args.prompt_id = args.exp_name  # train_model uses it to name its logging dir
    model = train_model(args, model, tokenizer, laundered, laundered, out_dir=args.out_dir)
    adapter_dir = os.path.join(args.out_dir, "adapter")
    model.save_pretrained(adapter_dir)
    print(f"Laundered data: {os.path.join(args.out_dir, 'laundered.json')}\nAdapter: {adapter_dir}")
    print(f"Next: python scripts/detect.py --model {args.model} --adapter {adapter_dir} --suspect <members.json> "
          f"--reference <nonmembers.json> --prefix <prefix.json>")


if __name__ == "__main__":
    main()
