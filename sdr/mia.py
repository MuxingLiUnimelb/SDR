# This implementation is adapted from Min-K% and WikiMIA: https://github.com/swj0419/detect-pretrain-code 
import os
from tqdm import tqdm
import torch 
import numpy as np
import torch
import zlib
from .metrics import *
import os
import torch.nn.functional as F
from transformers import set_seed
import torch 
import random
from accelerate import Accelerator
import torch 
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import copy
# from sentence_transformers import SentenceTransformer, util


def fix_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)

def to_input_ids(tokenizer, x, device):
    # 如果已经是 token id 列表
    if isinstance(x, (list, tuple)) and all(isinstance(t, int) for t in x):
        return torch.tensor([x], device=device)

    # 如果是多段字符串，拼成一段
    if isinstance(x, (list, tuple)) and all(isinstance(t, str) for t in x):
        x = " ".join(x)

    # 如果是 dict，常见键兜底
    if isinstance(x, dict):
        for k in ("text", "input", "content", "prompt"):
            if k in x:
                x = x[k]
                break

    # 其他类型统一转成字符串（避免 None / 数字 / ndarray / Series）
    if not isinstance(x, str):
        try:
            import numpy as np
            import pandas as pd
            if isinstance(x, (np.ndarray,)):
                x = " ".join(map(str, x.tolist()))
            elif isinstance(x, (pd.Series, pd.Array)):
                x = " ".join(map(str, list(x)))
            else:
                x = str(x)
        except Exception:
            x = str(x)

    enc = tokenizer(x, return_tensors="pt", truncation=True)
    return enc["input_ids"].to(device)

def get_ll(sentence, model, tokenizer, device):
    # to_input_ids 里面如果已经做了 tokenizer(...) 也没问题，这里统一保证 dtype 正确
    input_ids = to_input_ids(tokenizer, sentence, device)

    # 保证在正确的 device 且为 long 类型
    if isinstance(input_ids, dict):
        input_ids = input_ids["input_ids"]
    input_ids = input_ids.to(device).long()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return get_all_prob(input_ids, loss, logits)


def get_conditional_ll(input_text, target_text, model, tokenizer, device):
    input_encodings = tokenizer(input_text, return_tensors="pt")
    target_encodings = tokenizer(target_text, return_tensors="pt")

    # 拼接后转到 device，并强制为 long
    concat_ids = torch.cat(
        (input_encodings.input_ids, target_encodings.input_ids),
        dim=1
    ).to(device).long()

    labels = concat_ids.clone()
    labels[:, : input_encodings.input_ids.size(1)] = -100  # mask 前缀

    with torch.no_grad():
        outputs = model(input_ids=concat_ids, labels=labels)
    loss, logits = outputs[:2]
    return get_all_prob(labels, loss, logits)

# def get_ll(sentence, model, tokenizer, device):
#     # input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
#     input_ids = to_input_ids(tokenizer, sentence, device)
#     with torch.no_grad():
#         outputs = model(input_ids, labels=input_ids)
#     loss, logits = outputs[:2]
#     return get_all_prob(input_ids, loss, logits)

# def get_conditional_ll(input_text, target_text, model, tokenizer, device):
#     input_encodings = tokenizer(input_text, return_tensors="pt")
#     target_encodings = tokenizer(target_text, return_tensors="pt")
#     concat_ids = torch.cat((input_encodings.input_ids.to(device), target_encodings.input_ids.to(device)), dim=1)
#     labels = concat_ids.clone()
#     labels[:, : input_encodings.input_ids.size(1)] = -100
#     with torch.no_grad():
#         outputs = model(concat_ids, labels=labels)
#     loss, logits = outputs[:2]
#     return get_all_prob(labels, loss, logits)

def get_all_prob(input_ids, loss, logits):
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)
    ll = -loss.item()  # log-likelihood
    ppl = torch.exp(loss).item()
    prob = torch.exp(-loss).item()
    return prob, ll , ppl, all_prob, loss.item()


def inference(model1, model2, tokenizer1, tokenizer2, target_data, prefix, accelerator, num_shots, ex):
    # print("target_data:", target_data)
    pred = {}
    ll = get_ll(target_data, model1, tokenizer1,accelerator.device)[1]
    if int(num_shots) != 0:   
        ll_nonmember = get_conditional_ll(" ".join(prefix), target_data, model1, tokenizer1, accelerator.device)[1]
    pred["recall"] = ll_nonmember / ll

    # baselines 
    input_ids = torch.tensor(tokenizer1.encode(target_data), dtype=torch.long).unsqueeze(0).to(accelerator.device)
    with torch.no_grad():
        outputs = model1(input_ids, labels=input_ids)
    _, logits = outputs[:2]
    ll_ref = get_ll(target_data, model2, tokenizer2, accelerator.device)[1]

    # loss and zlib
    pred["ll"] = ll
    pred["ref"] = ll - ll_ref
    pred["perplexity"] = -get_ll(target_data, model1, tokenizer1,accelerator.device)[2]
    pred["zlib"] = ll / len(zlib.compress(bytes(target_data, "utf-8")))

    # For mink and mink++
    input_ids = input_ids[0][1:].unsqueeze(-1)
    probs = F.softmax(logits[0, :-1], dim=-1)
    log_probs = F.log_softmax(logits[0, :-1], dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
    mu = (probs * log_probs).sum(-1)
    sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)

    ## mink
    for ratio in [0.2]:
        k_length = int(len(token_log_probs) * ratio)
        topk = np.sort(token_log_probs.cpu())[:k_length]
        pred[f"mink_{ratio}"] = np.mean(topk).item()
        
    ## mink++
    mink_plus = (token_log_probs - mu) / sigma.sqrt()
    for ratio in [0.2]:
        k_length = int(len(mink_plus) * ratio)
        topk = np.sort(mink_plus.cpu())[:k_length]
        pred[f"mink++_{ratio}"] = np.mean(topk).item()

    ex["pred"] = pred
    return ex

def evaluate_data(test_data, model1, model2, tokenizer1, tokenizer2, prefix, accelerator, total_shots):
    all_output = []
    for ex in tqdm(test_data):
        prefix_copy = copy.deepcopy(prefix)
        new_ex = inference(model1, model2, tokenizer1, tokenizer2, ex["input"], prefix_copy, accelerator, total_shots, ex)
        # print("label:", new_ex["label"])
        all_output.append(new_ex)
    return all_output

def MIA_evaluate(args, training_dataset, model1, tokenizer1, model2, tokenizer2, nonmember_prefix, total_shots):
    accelerator = Accelerator()
    # evaluate the data
    all_output = evaluate_data(training_dataset, model1, model2, tokenizer1, tokenizer2, nonmember_prefix, accelerator, total_shots)
    
    # save the results
    all_output_path = os.path.join(args.out_dir, f"{args.dataset_name}", f"{args.model.split('/')[1]}_{args.ref_model.split('/')[1]}", f"{args.sub_data}", f"{args.prefix_size}_shot.json")
    os.makedirs(os.path.dirname(all_output_path), exist_ok=True)

    # evaluate the results
    result = fig_fpr_tpr(all_output, all_output_path)        
    return result
