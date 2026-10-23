"""Glue between the entry-point scripts and the original SDR code. No algorithm logic lives here."""
import argparse
import os

from tqdm import tqdm

from .llm_api import gpt_synthetic_prefix
from .mia import MIA_evaluate
from .sdr_loop import prompt_reverse_loop
from .utils import load_json, save_json

MIA_METRICS = ["ll", "zlib", "mink_0.2", "recall", "ref"]


def avg_auc(result):
    """The SDR score: mean AUC-ROC over loss, zlib, Min-K% (20%), ReCaLL and reference-model calibrated loss."""
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


def check_files(templates_path, register_prompts_path, adapter=None):
    """Validate detection inputs on disk before any model is loaded."""
    for name, path in (("templates", templates_path), ("register_prompts", register_prompts_path)):
        if not os.path.isfile(path):
            raise ValueError(f"--{name} file not found: {path}")
    n_templates, n_registers = len(load_json(templates_path)), len(load_json(register_prompts_path))
    if n_templates != n_registers:
        raise ValueError(f"--templates has {n_templates} entries but --register_prompts has {n_registers}; "
                         f"build the templates from the same register prompts")
    if adapter is not None:
        if os.path.isdir(adapter):
            if not os.path.isfile(os.path.join(adapter, "adapter_config.json")):
                raise ValueError(f"--adapter {adapter} has no adapter_config.json")
        elif adapter.count("/") != 1 or adapter.startswith((".", "/", "~")):
            raise ValueError(f"--adapter {adapter} is neither a local directory nor a Hugging Face repo id")


def work_dir_for(out_path):
    """Per-run working directory next to the result file (results/a.json -> results/a/)."""
    return os.path.splitext(os.path.abspath(out_path))[0]


def build_args(model, out_dir, prefix_size=7, dataset_name="custom", sub_data="default", ref_model="none/none", **extra):
    """Namespace carrying the attributes the original library code reads from `args`."""
    return argparse.Namespace(
        model=model if "/" in model else f"local/{model}",
        ref_model=ref_model if "/" in ref_model else f"local/{ref_model}",
        out_dir=out_dir, dataset_name=dataset_name, sub_data=sub_data, prefix_size=prefix_size, **extra)


def run_detection(model, tokenizer, ref_model, suspect, reference, prefix, templates_path, register_prompts_path,
                  out_dir, model_name, ref_model_name, rounds=10, l=5, m=5, prefix_size=7):
    """Baseline MIA on the original texts, then SDR (Stage 1 + `rounds` Stage-2 rounds).

    As in the paper, the reference model is fed the suspect model's tokenizer.
    Returns {inferred prompt: MIA results, ..., "original": baseline MIA results}.
    """
    os.makedirs(out_dir, exist_ok=True)
    suspect_path = os.path.join(out_dir, "suspect_texts.json")
    save_json(suspect, suspect_path)
    args = build_args(model_name, os.path.join(out_dir, "mia_logs"), prefix_size=prefix_size, ref_model=ref_model_name,
                      original_path=suspect_path, trigger_path=templates_path, domain_path=register_prompts_path)

    full_data = []
    for nm_data, m_data in zip(reference, suspect):
        full_data.append({"input": nm_data, "label": 0})
        full_data.append({"input": m_data, "label": 1})
    original = MIA_evaluate(args, full_data, model, tokenizer, ref_model, tokenizer, prefix, prefix_size)

    best_prompts = prompt_reverse_loop(args, suspect, reference, model, tokenizer, ref_model, tokenizer, prefix,
                                       rounds=rounds, l=l, m=m)
    best_prompts["original"] = original
    return best_prompts


def launder(texts, secret_prompt):
    """Rewrite every text with the secret prompt, as the data launderer would."""
    return [gpt_synthetic_prefix(t, secret_prompt) for t in tqdm(texts)]
