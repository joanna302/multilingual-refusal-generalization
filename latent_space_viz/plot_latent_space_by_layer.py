import torch
import json
import os
import argparse
import json
import numpy as np 
from pathlib import Path
from dotenv import load_dotenv
from transformers import pipeline, AutoModelForCausalLM
import pandas as pd

from utils.utils import load_and_sample_datasets, filter_data, get_refusal_scores_detector
from utils.process_activations import  get_activations, compute_and_plot_reduction_with_refusal, compute_and_plot_reduction_with_classifier

def parse_arguments():
    """Parse model path argument from command line."""
    load_dotenv("..", override=True)
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--model_lg', type=str, required=True, help='Language of the finetuning data')
    parser.add_argument('--refusal_detection_method', type=str, default='detector_v6')
    parser.add_argument('--checkpoint', type=str, default='checkpoint-3988')
    parser.add_argument('--prompts_type', type=str, default='all')
    return parser.parse_args()



def compute_and_plot_with_refusal(lg1, lg2, model_base, path, prompt_type, checkpoint, pca="refusal", layer=-1, lg_model="en"): 
    if prompt_type=="all": 
        harmful_train_vanilla_lg1, harmless_train_vanilla_lg1 = load_and_sample_datasets(lg1, "vanilla")
        harmful_train_adversarial_lg1, harmless_train_adversarial_lg1 = load_and_sample_datasets(lg1, "adversarial")
        harmful_train_lg1=pd.concat([harmful_train_vanilla_lg1, harmful_train_adversarial_lg1])
        harmless_train_lg1 = pd.concat([harmless_train_vanilla_lg1, harmless_train_adversarial_lg1])

        harmful_train_vanilla_lg2, harmless_train_vanilla_lg2 = load_and_sample_datasets(lg2, "vanilla")
        harmful_train_adversarial_lg2, harmless_train_adversarial_lg2 = load_and_sample_datasets(lg2, "adversarial")
        harmful_train_lg2=pd.concat([harmful_train_vanilla_lg2, harmful_train_adversarial_lg2])
        harmless_train_lg2 = pd.concat([harmless_train_vanilla_lg2, harmless_train_adversarial_lg2])
      
    else: 
        harmful_train_lg1, harmless_train_lg1 = load_and_sample_datasets(lg1, prompt_type)
        harmful_train_lg2, harmless_train_lg2 = load_and_sample_datasets(lg2, prompt_type)

    path_activations = f"{path}/activations/{checkpoint}/layer_{layer}"
    path_plot=f"{path}/{lg1}/{checkpoint}"

    if not os.path.exists(path_activations):
        os.makedirs(path_activations)

    if pca=="refusal": 
        if not os.path.exists(f"{path_activations}/activation_harmful_filtered_{lg1}.json"):  
            # filter refusal lg1 
            print("Compute activations refusal")
            if not os.path.exists(f"{path}/activations/{checkpoint}/harmful_train_lg1_filtered.json"):  
                harmful_train_lg1_filtered, harmless_train_lg1_filtered = filter_data(model_base, harmful_train_lg1, harmless_train_lg1, detector_model, refusal_detection_method="detector_v6")
                with open(f"{path}/activations/{checkpoint}/harmful_train_lg1_filtered.json", "w") as js :
                    json.dump(harmful_train_lg1_filtered, js)
                with open(f"{path}/activations/{checkpoint}/harmless_train_lg1_filtered.json", "w") as js :
                    json.dump(harmless_train_lg1_filtered, js)
            else : 
                with open(f"{path}/activations/{checkpoint}/harmful_train_lg1_filtered.json", "r") as js:
                    harmful_train_lg1_filtered = json.load(js)
                with open(f"{path}/activations/{checkpoint}/harmless_train_lg1_filtered.json", "r") as js :
                    harmless_train_lg1_filtered = json.load(js)
  
            print(f"Compute activations {lg1}")
            activation_harmful_lg1_filtered = get_activations(model_base.model, harmful_train_lg1_filtered,  model_base.tokenize_instructions_fn,  layer_idx=layer)
            activation_harmless_lg1_filtered = get_activations( model_base.model, harmless_train_lg1_filtered, model_base.tokenize_instructions_fn, layer_idx=layer)
            torch.save(activation_harmful_lg1_filtered, f"{path_activations}/activation_harmful_filtered_{lg1}.pt")
            torch.save(activation_harmless_lg1_filtered, f"{path_activations}/activation_harmless_filtered_{lg1}.pt")
        else : 
            print(f"Load activations {lg1}")
            activation_harmful_lg1_filtered= torch.load(f"{path_activations}/activation_harmful_filtered_{lg1}.pt")
            activation_harmless_lg1_filtered= torch.load(f"{path_activations}/activation_harmless_filtered_{lg1}.pt")
    elif pca=="harmful": 
        if not os.path.exists(f"{path_activations}/activation_harmful_{lg1}.pt"):  
            # filter refusal lg1 
            print(f"Compute activations {lg1}")
            activation_harmful_lg1= get_activations(model_base.model, harmful_train_lg1,  model_base.tokenize_instructions_fn, layer_idx=layer)
            activation_harmless_lg1 = get_activations( model_base.model, harmless_train_lg1, model_base.tokenize_instructions_fn,  layer_idx=layer)
            torch.save(activation_harmful_lg1, f"{path_activations}/activation_harmful_{lg1}.pt")
            torch.save(activation_harmless_lg1, f"{path_activations}/activation_harmless_{lg1}.pt")
        else : 
            print(f"Load activations {lg1}")
            activation_harmful_lg1= torch.load(f"{path_activations}/activation_harmful_{lg1}.pt")
            activation_harmless_lg1= torch.load(f"{path_activations}/activation_harmless_{lg1}.pt")

    if not os.path.exists(path_plot):
        os.makedirs(path_plot)

    if not os.path.exists(f"{path_activations}/activation_harmful_{lg2}.pt"):   
        print(f"Compute activations {lg2}")
        activation_harmful_lg2 = get_activations(model_base.model, harmful_train_lg2,  model_base.tokenize_instructions_fn, layer_idx=layer)
        activation_harmless_lg2 = get_activations( model_base.model, harmless_train_lg2, model_base.tokenize_instructions_fn, layer_idx=layer)
        torch.save(activation_harmful_lg2, f"{path_activations}/activation_harmful_{lg2}.pt")
        torch.save(activation_harmless_lg2, f"{path_activations}/activation_harmless_{lg2}.pt")
    else : 
        print(f"Load activations {lg2}")
        activation_harmful_lg2= torch.load(f"{path_activations}/activation_harmful_{lg2}.pt")
        activation_harmless_lg2= torch.load(f"{path_activations}/activation_harmless_{lg2}.pt")
    
    if not os.path.exists(f"{path_activations}/activation_refusal_{lg2}.pt"):   
        print(f"Compute activations refusal {lg2}")
        data_lg2 = pd.concat([harmful_train_lg2,harmless_train_lg2])
        label_data_lg2 = get_refusal_scores_detector(model_base.model, data_lg2, model_base.tokenize_instructions_fn, model_base.tokenizer, detector_model)
        mask = label_data_lg2.cpu().numpy().astype(bool)
        data_refusal = data_lg2[mask]
        data_nn_refusal = data_lg2[~mask]

        activation_refusal_lg2 = get_activations(model_base.model, data_refusal,  model_base.tokenize_instructions_fn, layer_idx=layer)
        activation_nn_refusal_lg2 = get_activations( model_base.model, data_nn_refusal, model_base.tokenize_instructions_fn, layer_idx=layer)
        torch.save(activation_refusal_lg2, f"{path_activations}/activation_refusal_{lg2}.pt")
        torch.save(activation_nn_refusal_lg2, f"{path_activations}/activation_non_refusal_{lg2}.pt")
    else : 
        print(f"Load activations {lg2}")
        activation_refusal_lg2= torch.load(f"{path_activations}/activation_refusal_{lg2}.pt")
        activation_nn_refusal_lg2= torch.load(f"{path_activations}/activation_non_refusal_{lg2}.pt")
    
    # compute refusal 
    if not os.path.exists(f"{path}/activations/{checkpoint}/refusal_labels_{lg2}.pt"):  
        print("Compute refusal labels")
        data_lg2 = pd.concat([harmful_train_lg2,harmless_train_lg2])
        label_data_lg2_refusal = get_refusal_scores_detector(model_base.model, data_lg2, model_base.tokenize_instructions_fn, model_base.tokenizer, detector_model)
        torch.save(label_data_lg2_refusal, f"{path}/activations/{checkpoint}/refusal_labels_{lg2}.pt")
    else : 
        print("Load refusal labels")
        label_data_lg2_refusal= torch.load(f"{path}/activations/{checkpoint}/refusal_labels_{lg2}.pt")

    #compute_and_plot_reduction_with_refusal(activation_harmful_lg1_filtered, activation_harmless_lg1_filtered, activation_refusal_lg2, activation_nn_refusal_lg2, activation_harmful_lg2, activation_harmless_lg2, path_plot, lg2, checkpoint=checkpoint, reduction_type='pca',  prompt_type=prompt_type)

    compute_and_plot_reduction_with_classifier(activation_harmful_lg1_filtered, activation_harmless_lg1_filtered, activation_harmful_lg2, activation_harmless_lg2, label_data_lg2_refusal, path_plot, lg2, checkpoint=checkpoint, reduction_type='pca',  prompt_type=prompt_type, classifier="regression", classifier_form_prompts=True, layer=layer, lg_model=lg_model)
    #compute_and_plot_reduction_with_classifier(activation_harmful_lg1, activation_harmless_lg1, activation_harmful_lg2, activation_harmless_lg2, label_data_lg2_refusal, path_plot, lg2, checkpoint=checkpoint, reduction_type='pca',  prompt_type=prompt_type, classifier="regression", classifier_form_prompts=True, pos_token=pos_token )


if __name__ == "__main__":
    args = parse_arguments()

    model_alias = os.path.basename(args.model_path)


    model_base = AutoModelForCausalLM.from_pretrained(args.model_path, subfolder=f"{args.checkpoint}").to('cuda') 
    num_layers = model_base.config.num_hidden_layers

    detector_model = pipeline("text-classification", model="joanna302/refusal_detector_v6") 

    path_to_save = f"results/{model_alias}/{args.prompts_type}"

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    #list_lg = ['de', 'bn', 'ar', 'jv', 'es', 'mk', 'sw', 'tt', 'fr', 'ja', 'pt', 'el', 'zh', 'en','da', 'lo', 'pag','mt']
    list_lg =['en', "fr", "pt", "pag", "mt", "ar", "lo", "bn", "el"]

    lg1 = "all"
    for lg2 in list_lg: 
       for l in range(-1, -num_layers, -2): 
            compute_and_plot_with_refusal(lg1,lg2, model_base, path_to_save, args.prompts_type, args.checkpoint, pca="refusal", layer=l, lg_model=args.lg_model)