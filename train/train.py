from unsloth import FastLanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from trl import SFTTrainer, SFTConfig, DPOConfig, DPOTrainer, ORPOConfig, ORPOTrainer, GRPOConfig, GRPOTrainer
from unsloth.chat_templates import train_on_responses_only
from unsloth import apply_chat_template
from datasets import Dataset, load_dataset, concatenate_datasets

import os 
import torch
import pandas as pd
import argparse 
import torch
import gc
import math 

# Clear cache before training
torch.cuda.empty_cache()
gc.collect()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_data_path', 
        type=str, 
        default="../data/datasets_train")
    parser.add_argument(
        '--name_data', 
        type=str, 
        default="en")
    parser.add_argument(
        '--hf_id', 
        type=str, 
        default="joanna303")
    parser.add_argument(
        '--lr', 
        type=float, 
        default=8e-5)
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42)
    parser.add_argument(
        '--training_type', 
        type=str, 
        default="SFT")
    parser.add_argument(
        '--add_alpaca', 
        type=bool, 
        default=True)
    parser.add_argument(
        '--alpaca_ratio', 
        type=float, 
        default=1)
    return parser.parse_args()


def apply_template(example):
    if example["instruction"]==None: 
        example["instruction"]=" "
    if example["chosen_response"]==None : 
        example["chosen_response"]=" "
    conversation = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["chosen_response"]}
    ]
    return {"conversation": conversation}

    
if __name__ == "__main__": 

    args = parse_arguments()

    # Data prep 

    data_train = Dataset.from_pandas(pd.read_csv(f"{args.train_data_path}/data_{args.name_data}.csv"))

    if args.add_alpaca==True: 
        print("add alpaca in english")
        alpaca_data_full = pd.read_json("hf://datasets/yahma/alpaca-cleaned/alpaca_data_cleaned.json")

        # keep only a part of the alpaca dataset
        N = len(alpaca_data_full) * args.alpaca_ratio
        print(f"N={N}")
        alpaca_data = alpaca_data_full.sample(n=math.ceil(N), replace=False, random_state=42)
        print(f"Size of alpaca dataset: {len(alpaca_data)}. {len(alpaca_data)/len(alpaca_data_full)}% of the initial dataset")

        # process instruction and input
        alpaca_data['instruction'] = alpaca_data.apply(lambda x : f"{x['instruction']}\n{x['input']}", axis=1)
        alpaca_data = Dataset.from_pandas(alpaca_data.rename({"output":"chosen_response"}, axis=1))

    if args.add_alpaca==True: 
        name = f"{args.model_name}-Base_{args.name_data}_alpaca_{args.alpaca_ratio}_part_{args.training_type}_{args.lr}"
    else : 
        name = f"{args.model_name}_{args.name_data}_{args.training_type}_{args.lr}"

    os.environ['WANDB_PROJECT'] = name
    
    model_base, tokenizer_base = FastLanguageModel.from_pretrained(
        model_name = f"swiss-ai/Apertus-8B-2509",
        max_seq_length = 2048,   # Context length - can be longer, but uses more memory
        load_in_4bit = False,     # 4bit uses much less memory
        load_in_8bit = False,    # A bit more accurate, uses 2x memory
        full_finetuning = False, 
        device_map='auto', 
    )
    del tokenizer_base
    torch.cuda.empty_cache()
    gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(f"unsloth/{args.model_name}-Instruct-2509")
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.padding_side = 'right' # padding to right (otherwise SFTTrainer shows warning)

    data_train =data_train.map(apply_template)

    if args.add_alpaca==True : 
        alpaca_data =alpaca_data.map(apply_template)
        data_train = concatenate_datasets([alpaca_data, data_train])

    text = tokenizer.apply_chat_template(list(data_train["conversation"]), 
                                        tokenize = False, 
                                        add_special_tokens=False, 
                                        enable_thinking=False)

    data = pd.DataFrame(text, columns=["text"])

    dataset = Dataset.from_pandas(data)

    model = FastLanguageModel.get_peft_model(
        model_base,
        r = 32,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 32,  # Best to choose alpha = rank or rank*2
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth" , # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,   # We support rank stabilized LoRA
        loftq_config = None,  # And LoftQ
    )

    print("Start SFT training")

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        use_gradient_checkpointing = "unsloth" , # True or "unsloth" for very long context
        args = SFTConfig(
            dataset_text_field = "text",
            per_device_train_batch_size = 16,
            #gradient_accumulation_steps = 4, # Use GA to mimic batch size!
            warmup_steps = 5,
            num_train_epochs = 3, # Set this for 1 full training run.
            max_steps = -1,  
            learning_rate = args.lr, # 2e-4 = high lr / 2e-5 = low lr / 8e-5 = middle lr 
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed=args.seed,
            data_seed=42, 
            report_to = "wandb", # Use this for WandB etc
            #eval_strategy="steps", 
            save_strategy="steps", 
            save_steps=1/6, 
            push_to_hub=True, 
            hub_model_id=f"{args.hf_id}/{name}", 
            hub_token=args.hf_token, 
            hub_strategy="all_checkpoints",
            padding_free=True,
        ),
    )

    # train model 
    trainer_stats = trainer.train()

    model.save_pretrained(name)  
    tokenizer.save_pretrained(name) 

    model.push_to_hub(f"{args.hf_id}/{name}", token = args.hf_token) # Online saving
    tokenizer.push_to_hub(f"{args.hf_id}/{name}", token = args.hf_token) # Online saving