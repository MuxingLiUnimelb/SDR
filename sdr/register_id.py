from tqdm import tqdm
import json
import numpy as np
import torch
from torch import nn
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from torch.nn.functional import softmax
import torch
import torch.nn.functional as F
from torch.nn.functional import softmax
from typing import List, Tuple
from .llm_api import get_completion, generate_prompt, gpt_synthetic_prefix

def get_first_sentence(text):
    for sep in [".", "?", "!"]:
        if sep in text:
            return text.split(sep)[0].strip() + sep
    return text.strip()

def get_domain(original_path, trigger_path, domain_path, model, tokenizer, m):
    with open(original_path, "r", encoding="utf-8") as f:
        training_data = json.load(f)
    questions = [get_first_sentence(data) for data in training_data[:m]]
    with open(trigger_path, "r", encoding="utf-8") as f:
        trigger_dict = json.load(f)
    print(f"Loaded {len(trigger_dict)} trigger phrases from dictionary")
    with open(domain_path, "r", encoding="utf-8") as f:
        domain_prompts = json.load(f)
    print(f"Loaded {len(domain_prompts)} domain prompts")
    trigger_candidates = []
    for key in trigger_dict.keys():
        single_trigger_list = trigger_dict[key]
        trigger_candidates.append(single_trigger_list)

    top_trigger_conf_scores = rank_triggers_by_confidence(model, tokenizer, trigger_candidates, questions, top_k=5)
    top_trigger_index = [trigger_candidates.index(trigger) for trigger, score in top_trigger_conf_scores]
    print(f"Top trigger phrases based on confidence scores: {top_trigger_index}")
    domain_prompts = [domain_prompts[i] for i in top_trigger_index]
    print(f"Selected domain prompts based on top triggers: {domain_prompts}")
    return domain_prompts

def full_text_builder(trigger, text):
    prompt = f"""
    You are an expert in natural language rewriting and template-based sentence construction.

    Your task:
    - Use the following template as the **exact structural guide**:  
    {trigger}
    - Replace placeholders like <ENTITY>, <PLACE>, <DATE>, <EVENT>, <DESCRIPTION> with information from the input text.
    - Preserve all important factual information from the input text.
    - Ensure **perfect English grammar** and natural flow in the rewritten sentence.
    - If the input does not contain certain details, keep the placeholder but make sure the sentence is still grammatical.

    Input text:
    {text}

    Output:
    Return **only** the rewritten sentence in the template format.
    """

    response = get_completion(prompt)
    return response.strip()

def compute_confidence_score(model, tokenizer, trigger: str, base_texts: List[str], max_length=512) -> List[float]:
    model.eval()
    results = []

    for text in base_texts:
        full_input = full_text_builder(trigger, text)
        input_ids = tokenizer.encode(full_input, return_tensors="pt", truncation=True, max_length=max_length).cuda()
        input_ids = input_ids.long()
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits  # shape: [1, seq_len, vocab_size]
            probs = F.softmax(logits, dim=-1)  # [1, seq_len, vocab_size]
            max_conf = probs.max(dim=-1).values.squeeze(0)  # [seq_len]
            avg_conf = max_conf.mean().item()  # average confidence
        results.append(avg_conf)
    return results

def rank_triggers_by_confidence(model, tokenizer, trigger_list: List[str], base_texts: List[str], top_k=5) -> List[Tuple[str, float]]:
    all_scores = []
    for one_trig in trigger_list:
        scores = compute_confidence_score(model, tokenizer, one_trig, base_texts)
        print(f"Trigger: {one_trig}, Scores: {scores}")
        avg_score = np.mean(scores)
        all_scores.append((one_trig, avg_score))
        
    ranked = sorted(all_scores, key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

