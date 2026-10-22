"""Offline step: build the register templates used by Stage 1 (register identification).

Adapted from trigger_part/training_reverser.py (reference rewrites) and
trigger_part/trigger_extractor_template.py (template abstraction).
"""
from tqdm import tqdm

from .llm_api import get_completion, gpt_synthetic_prefix


def get_unified_template(sentences):

    prompt = f"""
    You are an expert in syntax and linguistic abstraction.

    Given the following English sentences, ignore all specific names, dates, locations, and other concrete details.
    Abstract them into one single generalized English sentence structure (template) that can cover all of them,
    replacing variable elements with placeholders.

    Sentences:
    {chr(10).join(f"{i+1}. {s}" for i, s in enumerate(sentences))}

    Output only the generalized English template sentence.
    """

    response = get_completion(prompt)
    return response.strip()


def get_first_sentence(text):
    for sep in [".", "?", "!"]:
        if sep in text:
            return text.split(sep)[0].strip() + sep
    return text.strip()


def build_reference_rewrites(texts, register_prompts, gpt_model="gpt-4o-mini", n_texts=10):
    """Rewrite the first `n_texts` texts with every register prompt (training_reverser.py main)."""
    synthetic_list = []
    for p in tqdm(register_prompts):
        print(f"======== Using prompt:{p} ========")
        for data in tqdm(texts[:n_texts]):
            all_prompt = p + "Only return the rewritten content (without like Here's a ... reinterpretation/version of the text:). Emojes are not allowed."
            synthetic_list.append({"prompt": p, "input": data, "rewrite": gpt_synthetic_prefix(data, all_prompt, model=gpt_model)})
    return synthetic_list


def build_templates(reference_rewrites, n_texts=10):
    """Abstract the first sentences of each register's rewrites into one template (trigger_extractor_template.py)."""
    trigger_dictionary = {}
    for index in range(len(reference_rewrites) // n_texts):
        sentences = []
        for item in reference_rewrites[n_texts * index:n_texts * index + n_texts]:
            sentences.append(item["rewrite"])
        first_sentences = [get_first_sentence(s) for s in sentences]
        template = get_unified_template(first_sentences)
        print(f"Template for index {index}: {template}")
        trigger_dictionary[index] = template
    return trigger_dictionary
