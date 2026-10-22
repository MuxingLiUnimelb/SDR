import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD

from tqdm import tqdm
from datetime import datetime

from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
import datasets
from datasets import Dataset
from transformers import (
    set_seed,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)


os.environ['TOKENIZERS_PARALLELISM'] = "True"

def train_model(args, model, tokenizer, dataset_list, synthetic_non_member_list, out_dir=None):

    dataset = Dataset.from_dict({"text": dataset_list})
    eval_dataset = Dataset.from_dict({"text": synthetic_non_member_list})
    def tokenize_function(examples):
        tokenizer.pad_token = tokenizer.eos_token
        tokens = tokenizer(examples["text"], padding=True, truncation=True)
        return {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]}

    was_enabled = datasets.is_caching_enabled()
    datasets.disable_caching()
    tokens_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"], new_fingerprint="DO_NOT_ENABLE_CACHING", cache_file_name=None)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"], new_fingerprint="DO_NOT_ENABLE_CACHING", cache_file_name=None)
    
    def target_modules_exceptions(model_name):
        """
        Returns a list of target modules for LoRA, excluding the specified exceptions.
        """
        if model_name == "EleutherAI/pythia-6.9b":
            return ["query_key_value", "dense"]
        elif model_name == "meta-llama/Llama-2-7b-hf":
            return ["q_proj", "v_proj", "k_proj", "o_proj"]
        elif model_name == "mistralai/Mistral-7B-v0.3":
            return ["q_proj", "v_proj", "k_proj", "o_proj"]
        elif model_name == "tiiuae/falcon-7b":
            return ["query_key_value", "dense"]
        else:
            raise ValueError(f"Unsupported model architecture: {model_name}")
# 冻结所有模型参数
    for param in model.parameters():
        param.requires_grad = False
    # LoRA arguments
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # Adjust to your model's task, e.g., SEQ_2_SEQ_LM for seq-to-seq
        inference_mode=False,
        r=args.lora_dim,                           # Rank of the update matrices
        lora_alpha=args.lora_alpha,                # Scaling factor
        lora_dropout=0.1,                          # Dropout rate for LoRA
        target_modules=target_modules_exceptions(args.model),  # Target modules for LoRA
    )
    _model = get_peft_model(model, lora_config)
    # _model.half()
    _model.print_trainable_parameters()
    # Training arguments
    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    print("Using AdamW optimizer")
    if out_dir is None:
        training_args = TrainingArguments(
            output_dir=args.out_dir,
            eval_strategy="no",
            logging_strategy="epoch",
            learning_rate=args.lr,
            lr_scheduler_type='constant',
            per_device_train_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            weight_decay=0.1, 
            max_grad_norm=None,
            logging_dir=f'out/sft_output/prompt_{args.prompt_id}',
            report_to=[],
            run_name=args.exp_name,
            fp16=True,   
            # max_steps=args.epochs,  # Force only one optimization step per epoch
            # gradient_accumulation_steps=20,
            # warmup_steps=20,  # Warmup steps for learning rate scheduler
            # eta_min=1e-5,
            # save_strategy="epoch",
            save_total_limit=1,
            gradient_accumulation_steps=int(np.ceil(len(dataset) // (args.batch_size)))
        )
    else:
        training_args = TrainingArguments(
            output_dir=out_dir,
            eval_strategy="no",
            logging_strategy="epoch",
            learning_rate=args.lr,
            lr_scheduler_type='constant',
            per_device_train_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            weight_decay=0.1, 
            max_grad_norm=None,
            logging_dir=f'out/sft_output/prompt_{args.prompt_id}',
            report_to=[],
            run_name=args.exp_name,
            fp16=True,   
            # max_steps=args.epochs,  # Force only one optimization step per epoch
            # gradient_accumulation_steps=20,
            # warmup_steps=20,  # Warmup steps for learning rate scheduler
            # eta_min=1e-5,
            # save_strategy="epoch",
            save_total_limit=1,
            gradient_accumulation_steps=int(np.ceil(len(dataset) // (args.batch_size)))
        )

    optimizer = AdamW(_model.parameters(), lr=training_args.learning_rate)

    # Initialize the Trainer
    trainer = Trainer(
        model=_model,
        args=training_args,
        train_dataset=tokens_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        optimizers=(optimizer, None),
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    # Train the model
    trainer.train()
    # # Save the model
    # if not os.path.exists(args.out_dir):
    #     os.makedirs(args.out_dir)
    # _model.save_pretrained(args.out_dir)
    return _model