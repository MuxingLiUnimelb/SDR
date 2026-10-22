"""Glue between the entry-point scripts and the original SDR code. No algorithm logic lives here."""
import argparse
import os

from tqdm import tqdm

from .llm_api import gpt_synthetic_prefix
from .mia import MIA_evaluate
from .sdr_loop import prompt_reverse_loop
from .utils import save_json

MIA_METRICS = ["ll", "zlib", "mink_0.2", "recall"]


def avg_auc(result):
    """The SDR score: mean AUC-ROC over loss, zlib, Min-K% (20%) and ReCaLL."""
    return sum(result[k]["AUC-ROC"] for k in MIA_METRICS) / len(MIA_METRICS)


def summarize(result):
    """Return (baseline score, best SDR score, best inferred prompt) for one detection result."""
    baseline = avg_auc(result["original"])
    scores = {p: avg_auc(r) for p, r in result.items() if p != "original"}
    if not scores:
        return baseline, None, None
    best_prompt = max(scores, key=scores.get)
    return baseline, scores[best_prompt], best_prompt


def check_inputs(suspect, reference, prefix, min_texts=100):
    for name, texts in (("suspect", suspect), ("reference", reference), ("prefix", prefix)):
        if not isinstance(texts, list) or not all(isinstance(t, str) and t.strip() for t in texts):
            raise ValueError(f"{name} must be a JSON list of non-empty strings")
    if not prefix:
        raise ValueError("prefix needs at least one text (it is the ReCaLL prefix)")
    for name, texts in (("suspect", suspect), ("reference", reference)):
        if len(texts) < min_texts:
            raise ValueError(f"{name} needs at least {min_texts} texts (SDR scores each candidate prompt "
                             f"on {min_texts} suspect + {min_texts} reference texts); got {len(texts)}")
    if len(suspect) != len(reference):
        print(f"Warning: {len(suspect)} suspect vs {len(reference)} reference texts; "
              f"the baseline MIA pairs them and uses the first {min(len(suspect), len(reference))} of each.")


def build_args(model, out_dir, prefix_size=7, dataset_name="custom", sub_data="default", **extra):
    """Namespace carrying the attributes the original library code reads from `args`."""
    return argparse.Namespace(
        model=model if "/" in model else f"local/{model}",
        ref_model="none/none",  # reference-model metric is disabled; only used to build a log path
        out_dir=out_dir, dataset_name=dataset_name, sub_data=sub_data, prefix_size=prefix_size, **extra)


def run_detection(model, tokenizer, suspect, reference, prefix, templates_path, register_prompts_path,
                  out_dir, model_name, rounds=10, l=5, m=5, prefix_size=7):
    """Baseline MIA on the original texts, then SDR (Stage 1 + `rounds` Stage-2 rounds).

    Returns {inferred prompt: MIA results, ..., "original": baseline MIA results}.
    """
    os.makedirs(out_dir, exist_ok=True)
    suspect_path = os.path.join(out_dir, "suspect_texts.json")
    save_json(suspect, suspect_path)
    args = build_args(model_name, os.path.join(out_dir, "mia_logs"), prefix_size=prefix_size,
                      original_path=suspect_path, trigger_path=templates_path, domain_path=register_prompts_path)

    full_data = []
    for nm_data, m_data in zip(reference, suspect):
        full_data.append({"input": nm_data, "label": 0})
        full_data.append({"input": m_data, "label": 1})
    original = MIA_evaluate(args, full_data, model, tokenizer, None, None, prefix, prefix_size)

    best_prompts = prompt_reverse_loop(args, suspect, reference, model, tokenizer, None, None, prefix,
                                       rounds=rounds, l=l, m=m)
    best_prompts["original"] = original
    return best_prompts


def launder(texts, secret_prompt):
    """Rewrite every text with the secret prompt, as the data launderer would."""
    return [gpt_synthetic_prefix(t, secret_prompt) for t in tqdm(texts)]
