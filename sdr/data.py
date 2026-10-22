from datasets import load_dataset
import os
import random
import numpy as np
import json
   
def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for example in data:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

def create_dataset(dataset_name, sub_dataset_name, number_of_dataset, prfix_size=7):
    
    if dataset_name == "mimir_c4":
        dataset = load_dataset("iamgroot42/mimir", "c4", split="none", trust_remote_code=True)
        member_data = []
        nonmember_data = []
        for data in dataset:
            member_data.append(data["member"])
            nonmember_data.append(data["nonmember"])

    elif dataset_name == "mimir_wiki":  
        dataset = load_dataset("iamgroot42/mimir", "wikipedia_(en)", split=sub_dataset_name, trust_remote_code=True)
        member_data = []
        nonmember_data = []
        for data in dataset:
            member_data.append(data["member"])
            nonmember_data.append(data["nonmember"])

    elif dataset_name == "mimir_hackernews":
        dataset = load_dataset("iamgroot42/mimir", "hackernews", split=sub_dataset_name, trust_remote_code=True)
        member_data = []
        nonmember_data = []
        for data in dataset:
            member_data.append(data["member"])
            nonmember_data.append(data["nonmember"])

    else: 
        raise ValueError(f"Unknown dataset: {dataset_name}. Please modify the code to include the dataset. Make sure the dataset is in the same format.")

        # shuffle the datasets
    random.shuffle(member_data)
    random.shuffle(nonmember_data)
    nonmember_prefix= nonmember_data[-prfix_size:]
    num_shots = int(number_of_dataset)
    print(f"Number of member data: {len(member_data)}, number of nonmember data: {len(nonmember_data)}")
    nonmember_data = nonmember_data[:num_shots]
    mid = len(nonmember_data) // 2
    training_nonmember_data = nonmember_data[:mid]
    test_nonmember_data = nonmember_data[mid:]
    member_data = member_data[:num_shots]
    member_data_prefix = member_data[-prfix_size:]

    # full_data = [] 
    # # binary classification, the data need to be balanced. 
    # for nm_data, m_data in zip(nonmember_data, member_data):
    #     full_data.append({"input": nm_data, "label": 0})
    #     full_data.append({"input": m_data, "label": 1})

    return training_nonmember_data, test_nonmember_data, nonmember_prefix, member_data_prefix