"""Small helpers shared by the entry-point scripts."""
import json
import os
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def require_api_key():
    """Fail fast (before loading any model) when the OpenAI-compatible API is not configured."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Export it (and OPENAI_BASE_URL for an "
                         "OpenAI-compatible endpoint) before running this script.")


def load_model(model, model_path=None, adapter=None, use_float16=True, paper_dropout=False):
    """Load a causal LM the way the original drivers did, optionally attaching a LoRA adapter.

    `model` is the Hugging Face id (also used to pick LoRA target modules when training);
    `model_path` optionally points to local weights for the same model.
    `paper_dropout=True` reproduces the paper-era loading, where the LoRA dropout layers stayed in
    train mode until Stage 1 calls `model.eval()`; it therefore only affects the baseline MIA.
    """
    src = model_path or model
    llm = AutoModelForCausalLM.from_pretrained(src, return_dict=True, trust_remote_code=True,
        torch_dtype=torch.float16 if use_float16 else torch.float32,
        device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(src)
    if adapter is not None:
        from peft import PeftModel
        llm = PeftModel.from_pretrained(llm, adapter)
        if paper_dropout:
            for name, module in llm.named_modules():
                if "lora_dropout" in name:
                    module.train()
    return llm, tokenizer


def load_ref_model(ref_model, ref_model_path=None, use_float16=True):
    """Reference model for the `ref` metric (loaded like the paper-era drivers, fp16)."""
    return AutoModelForCausalLM.from_pretrained(ref_model_path or ref_model, return_dict=True, trust_remote_code=True,
        torch_dtype=torch.float16 if use_float16 else torch.float32,
        device_map="auto")
