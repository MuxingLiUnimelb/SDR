from .mia import MIA_evaluate
### Step 1: Generate initial prompts from x_orig
from .llm_api import get_completion, generate_prompt, gpt_synthetic_prefix
from tqdm import tqdm
import re
import random
from .register_id import get_domain
def infer_prompt(original_text, rewritten_text, domain_prompt, out_register = False):
    if not out_register:
        return f"""
        Given the following transformation task with a domain prompt, an original text, and a rewritten version, your job is to:

        **Keep the original domain prompt unchanged**, but **add a supplementary instruction sentence after it**. 
        The new instruction should guide how to rewrite the original text into the rewritten text, focusing on the key changes observed.

        This new sentence should:
        - Provide clear structural and stylistic guidance,
        - Be consistent with the original domain intent,
        - Help transform the original text into the rewritten version more effectively.

        Domain Prompt:
        \"\"\"{domain_prompt}\"\"\"

        Original Text:
        \"\"\"{original_text}\"\"\"

        Rewritten Text:
        \"\"\"{rewritten_text}\"\"\"

        Now only return whole new prompt(domain prompt + new instruction) without any explanation or additional text:
        """
    else:
        return f"""Given the following transformation task with a domain prompt, an original text, and a rewritten version, your job is to:
	    Analyze the differences between the original text and the rewritten text,
	    Modify the domain prompt accordingly so that it naturally encodes the transformation patterns observed,
	    The new domain prompt should provide clear structural and stylistic guidance consistent with the rewritten text, making it effective for reproducing such transformations.

        Domain Prompt:
        \"\"\"{domain_prompt}\"\"\"

        Original Text:
        \"\"\"{original_text}\"\"\"

        Rewritten Text:
        \"\"\"{rewritten_text}\"\"\"

        Now only return the modified domain prompt without any explanation or additional text."""

    
def extract_common_editing_prompt(prompt_list, new_best_prompt=None):
    prompt_block = "\n".join([f"- {p}" for p in prompt_list])
    prompt_section = (
        f'Current Prompt Suggestion:\n""" {new_best_prompt}. You may draw on it, but it is not required."""'
        if new_best_prompt else ''
    )
    meta_prompt = f"""
    You are an expert in rewriting prompt analysis.

    Below is a list of rewriting instructions. Your task is to **extract the common rewriting transformation** shared across all of them.

    Instructions:
    - Ignore surface wording differences and focus on their **overlapping intent**.
    - Summarize them into a **single abstract prompt** that could capture what all these prompts aim to do.
    - Do not explain or list features — just return the final abstract prompt.

    Here is the list of prompts:
    {prompt_block}

    {prompt_section}

    Output:
    [One abstract rewriting prompt capturing their shared intent]
    """
    return meta_prompt

def apply_prompt(prompt, x_orig):
    rewrite_prompt = f"""
    Below is an instruction:
    "{prompt}"
    Apply this instruction to the following text:
    "{x_orig}"
    Return only the rewritten version.
    """
    return rewrite_prompt


def split_prompts(text):
    lines = text.strip().splitlines()
    prompts = [re.sub(r"^\d+\.\s*", "", line).strip() for line in lines if line.strip()]
    return prompts


def update_from_regenerated(x_origs, model_after_training, tokenizer, domain_prompt, new_best_prompt=None):
    def get_first_sentence(text):
        words = text.split()
        return ' '.join(words[:30])
    possible_prompt = []
    for x_i in tqdm(x_origs):
        if new_best_prompt == None:
            domain_gpt_response = gpt_synthetic_prefix(x_i, domain_prompt)
        else:
            domain_gpt_response = gpt_synthetic_prefix(x_i, new_best_prompt)
        input_text = get_first_sentence(domain_gpt_response)
        input_tokens = tokenizer(input_text, return_tensors='pt').to(model_after_training.device)
        # input_tokens = input_tokens.long()
        outputs = model_after_training.generate(
            **input_tokens,
            max_new_tokens=512,  
            do_sample=True,      
            temperature=0.5,     # 提高多样性
            top_p=0.9,           # nucleus sampling
            repetition_penalty=1.05,  # 惩罚重复token
            pad_token_id=tokenizer.eos_token_id,  # 显式设置终止token
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        init_prompt = get_completion(infer_prompt(x_i, generated_text, domain_prompt))  # Replace with actual GPT API call
        print(f"Initial prompt for x_orig: {init_prompt}\n")
        possible_prompt.append(init_prompt)

    init_meta_prompt = extract_common_editing_prompt(possible_prompt)
    meta_prompt = get_completion(init_meta_prompt)
    meta_prompts = split_prompts(meta_prompt)
    print(f"Initial prompts generated from x_origs: {meta_prompts}")
    return meta_prompts
### NOTE: Replace all `mock_gpt(...)` with actual GPT API call logic

def MIA_scorer(one_step_x_origs, domain_prompt, one_step_x_non, model_after_training, tokenizer, ref_model, ref_tokenizer, nonmember_prefix, args):
    rewrite_list = []
    rewrite_non_member_list = []
    new_nonmember_prefix = []
    for x_orig in tqdm(one_step_x_origs):
        # gpt_response = gpt_synthetic_prefix(x_orig, domain_prompt)  # Replace with actual GPT API call
        rewrite_prompt = apply_prompt(domain_prompt, x_orig)
        gpt_response = get_completion(rewrite_prompt)  # Replace with actual GPT API call
        # print(f"Generated rewritten text for x_orig: {gpt_response}\n")
        rewrite_list.append(gpt_response)
    for non_member in tqdm(one_step_x_non):
        rewrite_prompt = apply_prompt(domain_prompt, non_member)
        gpt_response = get_completion(rewrite_prompt)
        # gpt_response = gpt_synthetic_prefix(non_member, domain_prompt)
        rewrite_non_member_list.append(gpt_response)
    for non_member_prefix in tqdm(nonmember_prefix):
        rewrite_prompt = apply_prompt(domain_prompt, non_member_prefix)
        gpt_response = get_completion(rewrite_prompt)
        # gpt_response = gpt_synthetic_prefix(non_member_prefix, domain_prompt)
        new_nonmember_prefix.append(gpt_response)
    full_data = []
    for nm_data, m_data in zip(rewrite_non_member_list, rewrite_list):
        full_data.append({"input": nm_data, "label": 0})
        full_data.append({"input": m_data, "label": 1})
    results = MIA_evaluate(args, full_data, model_after_training, tokenizer, ref_model, ref_tokenizer, new_nonmember_prefix, args.prefix_size)
    avg_score = (results["ll"]["AUC-ROC"]+ results["mink_0.2"]["AUC-ROC"] +  results["recall"]["AUC-ROC"]+results["ref"]["AUC-ROC"]+results["zlib"]["AUC-ROC"])/5
    return avg_score, results

def prompt_reverse_loop(args, x_origs, test_nonmember_data, model_after_training, tokenizer, ref_model, ref_tokenizer, nonmember_prefix, rounds=10, l = 5, m=5):
    print("==========GPT Response========")
    domain_list = get_domain(args.original_path, args.trigger_path, args.domain_path, model_after_training, tokenizer, m=m)
    prompt_score = {}
    domain_score = []
    real_domain_prompt = ""
    for domain_prompt in domain_list:
        one_step_x_origs = random.sample(x_origs, 20)  
        one_step_x_non = random.sample(test_nonmember_data, 20)     # Use a subset for efficiency
        avg_score = MIA_scorer(one_step_x_origs, domain_prompt, one_step_x_non, model_after_training, tokenizer, ref_model, ref_tokenizer, nonmember_prefix, args)
        domain_score.append(avg_score)
    real_domain_prompt = domain_list[domain_score.index(max(domain_score))]
    # real_domain_prompt = domain_list[4]
    # real_domain_prompt = "Rewrite the text as advice or guidance on a specific topic."
    one_step_x_origs = random.sample(x_origs, 100)  
    one_step_x_non = random.sample(test_nonmember_data, 100) 
    avg_score, results = MIA_scorer(one_step_x_origs, real_domain_prompt, one_step_x_non, model_after_training, tokenizer, ref_model, ref_tokenizer, nonmember_prefix, args)
    print(f"Best domain prompt: {real_domain_prompt}")
    print(f"========most possible domain prompt: {real_domain_prompt} ========")
    best_score = avg_score
    best_prompt = None
    for r in range(rounds):
        one_step_x_origs = random.sample(x_origs, l) 
        possible_prompt = update_from_regenerated(one_step_x_origs, model_after_training, tokenizer, domain_prompt=real_domain_prompt, new_best_prompt=best_prompt)
        one_step_x_origs = random.sample(x_origs, 20)  
        one_step_x_non = random.sample(test_nonmember_data, 20)     # Use a subset for efficiency
        avg_score, results = MIA_scorer(one_step_x_origs, possible_prompt, one_step_x_non, model_after_training, tokenizer, ref_model, ref_tokenizer, nonmember_prefix, args)
        # speed up process (reduce gpt calls)
        if avg_score > 0.65:
            one_step_x_origs = random.sample(x_origs, 100)  
            one_step_x_non = random.sample(test_nonmember_data, 100)     # Use a subset for efficiency
            avg_score, results = MIA_scorer(one_step_x_origs, possible_prompt, one_step_x_non, model_after_training, tokenizer, ref_model, ref_tokenizer, nonmember_prefix, args)
            if avg_score > best_score:
                best_score = avg_score
                best_prompt = possible_prompt
                print(f"Round {r+1}: New best prompt found with score {best_score}: {best_prompt}")
            prompt_score[possible_prompt[0]] = results
    return prompt_score