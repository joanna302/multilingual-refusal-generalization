import random 
import re 
import torch 
import pandas as pd 
import os 

from ...eval.eval_results import extract_generated_sentence
from utils.hook_utils import add_hooks, get_activation_addition_input_pre_hook, get_direction_ablation_input_pre_hook, get_direction_ablation_output_hook

PATH = "{root_path}/{prompts_type}/{data_type}/data_{harmtype}_{lg}.csv"

def load_harmful_harmless_datasets(lg=False, prompts_type="vanilla"):
    harmful_data = load_dataset(harmtype='harmful', instructions_only=True, lg=lg, data_type='train', prompts_type=prompts_type)
    harmless_data = load_dataset(harmtype='harmless', instructions_only=True, lg=lg, data_type='train',  prompts_type=prompts_type)
    return harmful_data, harmless_data

def load_dataset(harmtype: str, lg='all', data_type='train', prompts_type="vanilla", path="data"):

    file_path = PATH.format(root_path=path, data_type=data_type, harmtype=harmtype, lg=lg, prompts_type=prompts_type)
    dataset = pd.read_csv(file_path)
    if lg=="all":
        dataset = dataset.groupby('language').sample(30, random_state=42)

    return dataset

def filter_data(model_base, harmful_train, harmless_train, detector_model):
    """
    Filter datasets based on refusal scores.
    """

    print("Filtering train dataset")
    print(f"Number of harmful examples: {len(harmful_train)}")
    print(f"Number of harmless examples: {len(harmless_train)}")

    harmful_train_scores = get_refusal_scores_detector(model_base.model, harmful_train, model_base.tokenize_instructions_fn, model_base.tokenizer, detector_model)
    harmless_train_scores = get_refusal_scores_detector(model_base.model, harmless_train, model_base.tokenize_instructions_fn, model_base.tokenizer, detector_model)
    
    print(len([score for score in harmful_train_scores.tolist() if score == 1]))
    print(len([score for score in harmless_train_scores.tolist() if score == 0]))

    nb_min = min(len([score for score in harmless_train_scores.tolist() if score == 0]), len([score for score in harmful_train_scores.tolist() if score == 1]))

    harmful_train = [inst for inst, score in zip(harmful_train, harmful_train_scores.tolist()) if score==1][:nb_min]
    harmless_train = [inst for inst, score in zip(harmless_train, harmless_train_scores.tolist()) if score==0][:nb_min]
    
    print(f"Filtered {len(harmful_train)} harmful examples and {len(harmless_train)} harmless examples")

    return harmful_train, harmless_train


def get_refusal_scores_detector(model, instructions, tokenize_instructions_fn, tokenizer, model_detection, fwd_pre_hooks=[], fwd_hooks=[], batch_size=16):
 
    refusal_scores = torch.zeros(len(instructions), device=model.device)

    for i in range(0, len(instructions), batch_size):
        tokenized_instructions = tokenize_instructions_fn(instructions=instructions[i:i+batch_size]).to(model.device)
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            generated_ids = model.generate(
                input_ids=tokenized_instructions.input_ids.to(model.device),
                attention_mask=tokenized_instructions.attention_mask.to(model.device),
                max_new_tokens=25,
                temperature=0.7, top_p=0.8, top_k=20
            )
            output_ids_list = [generated_ids[i][len(tokenized_instructions.input_ids[i]):].tolist() for i in range(len(generated_ids))]

            text_generated = [tokenizer.decode(output_ids, skip_special_tokens=True) for output_ids in output_ids_list]

            cleaned_text = [extract_generated_sentence(x) for x in text_generated]

        outputs_refusal = model_detection(cleaned_text)
        
        labels = [int(out['label'].split('_')[-1]) for out in outputs_refusal]
     
        refusal_scores[i:i+batch_size] = torch.FloatTensor([0 if (label==2 or label==3) else label for label in labels])

    print(refusal_scores)

    return refusal_scores