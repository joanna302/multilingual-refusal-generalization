import torch
import json
import os
import argparse
import json
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

from peft import PeftModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from dotenv import load_dotenv
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from mpl_toolkits.mplot3d import Axes3D

from utils.utils import load_harmful_harmless_datasets, filter_data, get_refusal_scores_detector
from utils.process_activations import  get_activations, compute_and_plot_reduction_with_refusal, compute_and_plot_reduction_with_classifier
from utils.apertus_model import ApertusModel

def parse_arguments():
    """Parse model path argument from command line."""
    load_dotenv("..", override=True)
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--model_path_lora', type=str, required=False, help='Path to the lora model')
    parser.add_argument('--checkpoint', type=str, default='checkpoint-3988')
    parser.add_argument('--prompts_type', type=str, default='vanilla')
    return parser.parse_args()

def compute_and_plot_pca(activations, list_lg, languages, prompt_types, type="harmfulness"): 

    scaler = StandardScaler()
    activations_scaled = scaler.fit_transform(activations)

    if type=="harmfulness": 
        prompt_type_list = ['harmful', 'harmless']
    else : 
        prompt_type_list = ['refusal', 'non refusal']
    
    markers = {prompt_type_list[0]: 'X', prompt_type_list[1]: 'o'}

    # Step 2: Perform PCA
    n_components = min(50, activations.shape[0], activations.shape[1])  # or specify desired number
    pca = PCA(n_components=n_components)
    activations_pca = pca.fit_transform(activations_scaled)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.tab20(np.linspace(0, 1, len(list_lg)))

    for i, lang in enumerate(list_lg):
        for prompt_type in prompt_type_list:
            mask = (languages == lang) & (prompt_types == prompt_type)
            ax.scatter(activations_pca[mask, 0], 
                    activations_pca[mask, 1],
                    activations_pca[mask, 2],
                    c=[colors[i]], 
                    marker=markers[prompt_type],
                    label=f'{lang} ({prompt_type})' if i < 2 else '',
                    alpha=0.6,
                    edgecolors='black' if prompt_type == 'harmful' else 'none',
                    linewidths=0.5)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})')
    ax.set_title(f'PCA: {prompt_type_list[0]} (X) vs {prompt_type_list[1]} (O) by Language (3D)')
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('pca_by_language_and_type_3d.png', dpi=300, bbox_inches='tight')
    plt.show()

def compute_and_plot(model_base, path, prompts_type, checkpoint, list_lg, layer=-1): 

    activations_harmful=[]
    activations_harmless=[]

    activations_refusal=[]
    activations_non_refusal=[]

    for lg in list_lg : 
        harmful_data, harmless_data = load_harmful_harmless_datasets(lg, prompts_type)
   
        path_activations = f"{path}/activations/{checkpoint}/{lg}/{prompts_type}"
        path_plot=f"{path}/{lg}/plot/{checkpoint}/{prompts_type}"

        if not os.path.exists(path_plot):
            os.makedirs(path_plot)
    
        if not os.path.exists(path_activations):
            os.makedirs(path_activations)

        if not os.path.exists(f"{path_activations}/activation_harmful_{lg}.pt"):  
            print(f"Compute activations {lg}")
            activation_harmful_lg = get_activations(model_base.model, harmful_data, model_base.tokenize_instructions_fn, prompts_type)
            activation_harmless_lg = get_activations(model_base.model, harmless_data, model_base.tokenize_instructions_fn, prompts_type)
            torch.save(activation_harmful_lg, f"{path_activations}/activation_harmful_{lg}.pt")
            torch.save(activation_harmless_lg, f"{path_activations}/activation_harmless_{lg}.pt")

        else : 
            print(f"Load activations {lg}")
            activation_harmful_lg= torch.load(f"{path_activations}/activation_harmful_{lg}.pt")
            activation_harmless_lg= torch.load(f"{path_activations}/activation_harmless_{lg}.pt")
        
        activations_harmful.append(activation_harmful_lg)
        activations_harmless.append(activation_harmless_lg)
        
        ## possibility to compute the pca regarding to the harmfulness or the refusal

        if not os.path.exists(f"{path_activations}/activation_harmful_filtered_{lg}.pt"):  
            # filter refusal lg1 
            print("Compute activations refusal")
            if not os.path.exists(f"{path}/activations/{checkpoint}/harmful_data_{lg}_filtered.json"):  
                print("Filter data")
                harmful_data_filtered, harmless_data_filtered = filter_data(model_base, model_base.tokenize_instructions_fn, harmful_data, harmless_data, detector_model)
                with open(f"{path_activations}/harmful_data_{lg}_filtered.json", "w") as js :
                    json.dump(harmful_data_filtered, js)
                with open(f"{path_activations}/harmless_data_{lg}_filtered.json", "w") as js :
                    json.dump(harmless_data_filtered, js)
            else : 
                with open(f"{path_activations}/harmful_data_{lg}_filtered.json", "r") as js:
                    harmful_data_filtered = json.load(js)
                with open(f"{path_activations}/harmless_data_{lg}_filtered.json", "r") as js :
                    harmless_data_filtered = json.load(js)

            print(f"Compute activations {lg}")
            activation_harmful_lg_filtered = get_activations(model_base.model, harmful_data_filtered,  model_base.tokenize_instructions_fn,  layer_idx=layer)
            activation_harmless_lg_filtered = get_activations( model_base.model, harmless_data_filtered, model_base.tokenize_instructions_fn, layer_idx=layer)
            torch.save(activation_harmful_lg_filtered, f"{path_activations}/activation_harmful_filtered_{lg}.pt")
            torch.save(activation_harmless_lg_filtered, f"{path_activations}/activation_harmless_filtered_{lg}.pt")
        else : 
            print(f"Load activations {lg}")
            activation_harmful_lg_filtered= torch.load(f"{path_activations}/activation_harmful_filtered_{lg}.pt")
            activation_harmless_lg_filtered= torch.load(f"{path_activations}/activation_harmless_filtered_{lg}.pt")

        activations_refusal.append(activation_harmful_lg_filtered)
        activations_non_refusal.append(activation_harmless_lg_filtered)

    activations_harmful = np.vstack(activations_harmful)  
    activations_harmless = np.vstack(activations_harmless)  

    activations_harmfulness = np.vstack([activations_harmful, activations_harmless])

    activations_refusal=np.vstack(activations_refusal)
    activations_non_refusal=np.vstack(activations_non_refusal)

    activations_refusal_type = np.vstack([activations_refusal, activations_non_refusal])

    num_harmful_per_lang = activations_harmful.shape[0] // len(list_lg)
    num_harmless_per_lang = activations_harmless.shape[0] // len(list_lg)

    languages_labels = np.concatenate([
        np.repeat(list_lg, num_harmful_per_lang),  # Languages for harmful
        np.repeat(list_lg, num_harmless_per_lang)  # Languages for harmless
    ])

    harmfulness_labels = np.concatenate([
        np.repeat('harmful', activations_harmful.shape[0]),
        np.repeat('harmless', activations_harmless.shape[0])
    ])

    refusal_labels = np.concatenate([
        np.repeat('refusal', activations_refusal.shape[0]),
        np.repeat('non refusal', activations_non_refusal.shape[0])
    ])

    compute_and_plot_pca(activations_harmfulness, list_lg, languages_labels, harmfulness_labels, type="harmfulness")
    compute_and_plot_pca(activations_refusal_type, list_lg, languages_labels, refusal_labels, type="refusal")

if __name__ == "__main__":
    args = parse_arguments()

    model_alias = os.path.basename(args.model_path)

    #model_base = AutoModelForCausalLM.from_pretrained(args.model_path, subfolder=f"{args.checkpoint}").to('cuda') 
    model = ApertusModel(args.model_path, args.model_path_lora, args.checkpoint) 

    detector_model = pipeline("text-classification", model="joanna302/refusal_detector_v6") 

    path_to_save = f"results_lg/{model_alias}/{args.prompts_type}"

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    list_lg = ['de', 'bn', 'en']

    #list_lg = ['de', 'bn', 'ar', 'jv', 'es', 'mk', 'sw', 'tt', 'fr', 'ja', 'pt', 'el', 'zh', 'en','da', 'lo', 'pag','mt']

    compute_and_plot(model, path_to_save, args.prompts_type, args.checkpoint, list_lg, layer=-1)