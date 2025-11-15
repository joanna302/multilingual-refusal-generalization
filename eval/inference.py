from unsloth import FastLanguageModel
import datasets
import pandas as pd 
import torch 
import argparse 
from tqdm import tqdm 
import os 

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from torch.utils.data import DataLoader
from datasets import concatenate_datasets

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch._dynamo
torch._dynamo.config.cache_size_limit = 256 


def chat_template(example):
    prompt_type = "prompt"
    conversation = [
        {"role": "user", "content": example[prompt_type]}
    ]
    return {"conversation": conversation}

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name', 
        type=str, 
        default="Apertus-8B")
    parser.add_argument(
        '--data_eval_path', 
        type=str, 
        default="../data/datasets_eval/data_eval_all.csv")
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        default="checkpoint-3988")
    parser.add_argument(
        '--name_data', 
        type=str, 
        default="en")
    parser.add_argument(
        '--lr', 
        type=int, 
        default=8e-5)
    parser.add_argument(
        '--training_type', 
        type=str, 
        default="SFT")
    parser.add_argument(
        '--alpaca_ratio', 
        type=float, 
        default=1)
    parser.add_argument(
        '--add_alpaca', 
        type=bool, 
        default=True)
    parser.add_argument(
        '--repo_id', 
        type=str)
    return parser.parse_args()
    

def preprocess_data(data_eval): 
    print(set(data_eval['language']))
    data_low_bs = data_eval[data_eval['language'].apply(lambda x : True if x in list_lg_diff else False)]
    data_high_bs = data_eval[data_eval['language'].apply(lambda x : False if x in list_lg_diff else True )]
    data_high_bs = Dataset.from_pandas(data_high_bs).map(chat_template)
    data_low_bs = Dataset.from_pandas(data_low_bs).map(chat_template)
    return data_high_bs, data_low_bs

if __name__ == "__main__": 
    args = parse_arguments()

    list_lg_diff = ["lo", "tt", "ar", "el", "bn", "pag"]

    data_eval = pd.read_csv(args.data_eval_path)
    col_to_remove = ["focus","note"]
    for col in col_to_remove: 
        if col in data_eval.columns: 
            data_eval = data_eval.drop(col, axis=1)

    if args.add_alpaca==True: 
        model_name = f"{args.model_name}-Base_{args.name_data}_alpaca_{args.alpaca_ratio}_part_{args.training_type}_LoRA_{args.lr}_wo_pf"
    else : 
        model_name = f"{args.model_name}-Base_{args.name_data}_{args.training_type}_LoRA_{args.lr}"

    #model = AutoModelForCausalLM.from_pretrained(f"{args.repo_id}/{model_name}", subfolder=f"{args.checkpoint}").to('cuda') 
    #tokenizer = AutoTokenizer.from_pretrained(f"{args.repo_id}/{model_name}")

    max_seq_length = 2048 

    print(f"{args.repo_id}/{model_name}")

    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="swiss-ai/Apertus-8B-2509",  # Base model
        max_seq_length=max_seq_length,
        load_in_4bit=False, 
        dtype=None )

    # Load LoRA adapter directly
    model = PeftModel.from_pretrained(
        model,
        f"{args.repo_id}/{model_name}",
        subfolder=args.checkpoint if args.checkpoint else None
    )

    # Merge adapter weights for faster inference (optional)
    model = model.merge_and_unload()

    # Enable inference mode
    FastLanguageModel.for_inference(model)

    tokenizer.padding_side = 'left'

    tokenizer.eos_token = "<s>"
    tokenizer.pad_token = "<s>" 
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    ### process data
    data_high_bs, data_low_bs = preprocess_data(data_eval)

    text_low = tokenizer.apply_chat_template(
        list(data_low_bs['conversation']),
        tokenize = False,
        add_generation_prompt = True, # Must add for generation
        enable_thinking = False, # Disable thinking
    )

    data_low_bs = data_low_bs.add_column("text", text_low) 

    text_high = tokenizer.apply_chat_template(
        list(data_high_bs['conversation']),
        tokenize = False,
        add_generation_prompt=True, # Must add for generation
        enable_thinking = False, # Disable thinking
    )

    data_high_bs = data_high_bs.add_column("text", text_high)   

    bs_high = 32
    bs_low = 4

    data_loader_high_bs = DataLoader(data_high_bs, batch_size=bs_high, shuffle=False)
    data_loader_low_bs = DataLoader(data_low_bs, batch_size=bs_low, shuffle=False)
    print("dataloader ok")

    ### eval 
    dist_checkpoint = 0 
    outputs_list=[]
    distance_list=[]

    file_name = f"data_with_output_{model_name}_{args.checkpoint}.csv"
        
    model.eval()

    with torch.no_grad():
    
        for batch in tqdm(data_loader_high_bs):
            print(set(batch['language']))

            model_inputs = tokenizer(
                batch["text"], 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
            ).to(model.device)

            # conduct text completion
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=70,
                temperature = 0.7, 
                top_p = 0.8, 
                top_k = 20, 
                eos_token_id=tokenizer.eos_token_id
            )

            output_ids_list=[generated_ids[i][len(model_inputs.input_ids[i]):].tolist() for i in range(len(generated_ids))]
            output_ids_list = [[token for token in ids if token != 1] for ids in output_ids_list]
            output = [tokenizer.decode(output_ids) for output_ids in output_ids_list]
            print(output)

            outputs_list.extend(output)

        data_high_bs = data_high_bs.add_column("output", outputs_list) 

        outputs_list=[]

        for batch in tqdm(data_loader_low_bs):
            print(set(batch['language']))
            model_inputs = tokenizer(
                batch["text"], 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
            ).to(model.device)

            # conduct text completion
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=100,
                temperature = 0.7, 
                top_p = 0.8, 
                top_k = 20, 
                eos_token_id=tokenizer.eos_token_id
            )

            output_ids_list = [generated_ids[i][len(model_inputs.input_ids[i]):].tolist() for i in range(len(generated_ids))]
            output_ids_list = [[token for token in ids if token != 1] for ids in output_ids_list]
            output = [tokenizer.decode(output_ids) for output_ids in output_ids_list if output_ids != 0]

            print(output)

            outputs_list.extend(output)

        data_low_bs = data_low_bs.add_column("output", outputs_list)


        data_res = datasets.concatenate_datasets([data_high_bs, data_low_bs])

        data_res.to_csv(f"../results/Apertus-8B/{args.name_data}/{file_name}")
