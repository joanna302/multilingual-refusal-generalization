import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
import json
import numpy as np 
from scipy.spatial.distance import cdist
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

def chat_template(example):
    prompt_type = "vanilla"
    print(example[prompt_type])
    conversation = [
        {"role": "user", "content": example[prompt_type]}
    ]
    return {"conversation": conversation}

def get_activations(model, instructions, tokenize_instructions_fn, prompt_type="vanilla", batch_size=32, layer_idx=-1, token_idx=-1):
    last_activations= None

    instructions=list(instructions[prompt_type])

    for i in range(0, len(instructions), batch_size):
        tokenized_instructions = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])

        with torch.no_grad(): 
            outputs = model(
                input_ids=tokenized_instructions.input_ids.to(model.device),
                attention_mask=tokenized_instructions.attention_mask.to(model.device),
                output_hidden_states=True
            )

        activations = outputs.hidden_states[layer_idx][:, token_idx, :] 

        if last_activations is None:
            last_activations = activations
        else:
            last_activations = torch.cat((last_activations, activations), dim=0)

    return last_activations


def compute_and_plot_reduction(act_refusal, act_nn_refusal, dir, lg1, dist, reduction_type='pca', list_lg=[], n_compo=2, colors_type='cat'): 
    activations_ref_np = act_refusal.detach().cpu().to(torch.float16)#.numpy()
    activations_nn_ref_np = act_nn_refusal.detach().cpu().to(torch.float16)#.numpy()

    X = torch.cat([activations_ref_np, activations_nn_ref_np], dim=0).numpy()
    labels = [dist[0]] * activations_ref_np.shape[0] + [dist[1]] * activations_nn_ref_np.shape[0]

    label_to_num = {dist[0]: 0, dist[1]: 1}
    y_numeric = np.array([label_to_num[label] for label in labels]) 

    if reduction_type=='pca': 
        pca = PCA(n_components=n_compo, random_state=42)
        X_embedded = pca.fit_transform(X)
    elif reduction_type=='tsne': 
        X_embedded = TSNE(n_components=n_compo, perplexity=30, random_state=42).fit_transform(X)
    elif reduction_type=='umap-supervised': 
        X_embedded = umap.UMAP(random_state=42).fit_transform(X, y=y_numeric)
    elif reduction_type=='umap-unsupervised': 
        X_embedded = umap.UMAP(random_state=42).fit_transform(X)


    plt.figure(figsize=(10, 7)) 

    colors = {dist[0]: 'red', dist[1]: 'blue'}
    markers = {dist[0]: 'o', dist[1]: 's'}

    if list_lg==[]: 
        for i, (point, label) in enumerate(zip(X_embedded, labels)):
            plt.scatter(point[0], point[1], color=colors[label],  alpha=0.7, label=f"{label}")
    else: 
        if colors_type=='cat': 
            colors = {'de':'lightcoral', 'bn':"maroon", 'ar':"peachpuff", 'jv':"deeppink", 'es':"khaki", 'mk':"deeppink", 'sw':"deeppink", 'tt':"deeppink", 'fr':"turquoise", 'ja':"teal", 'pt':"cyan", 'el':"steelblue", 'zh':"lavender", 'en':"navy", 'da':"blueviolet", 'lo':"deeppink", 'pag':"deeppink", 'mt':"deeppink"}
        elif colors_type=='lg':
            colors = {'de':'lightcoral', 'bn':"maroon", 'ar':"peachpuff", 'jv':"darkorange", 'es':"khaki", 'mk':"olive", 'sw':"yellow", 'tt':"lightgreen", 'fr':"turquoise", 'ja':"teal", 'pt':"cyan", 'el':"steelblue", 'zh':"lavender", 'en':"navy", 'da':"blueviolet", 'lo':"plum", 'pag':"fuchsia", 'mt':"deeppink"}
        for i, (point, label, lg) in enumerate(zip(X_embedded, labels, list_lg+list_lg)):
            plt.scatter(point[0], point[1], marker=markers[label], color=colors[lg], alpha=0.7)

        # Unicize legend
        
        lang_handles = []
        for c in colors.values() :
            lang_handles.append(plt.scatter([], [], c=c, 
                                        marker='o', s=40, alpha=0.8, 
                                        edgecolors='black', linewidth=0.3))
        
        
        # Add legends with smaller font and potentially multiple columns

        lang_legend = plt.legend(lang_handles, colors.keys(), 
                                title='Languages', loc='upper left', 
                                bbox_to_anchor=(1.02, 1), fontsize=9)
        
        plt.gca().add_artist(lang_legend)

     # Composition legend (markers)
    comp_handles = []
    for m in markers.values() :
        comp_handles.append(plt.scatter([], [], c='gray', 
                                      marker=m, 
                                      s=40, alpha=0.8, 
                                      edgecolors='black', linewidth=0.3))
    
    comp_legend = plt.legend(comp_handles, markers.keys(), 
                           title='Types', loc='upper left', 
                           bbox_to_anchor=(1.02, 0.3), fontsize=9)
    

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    #plt.legend(by_label.values(), by_label.keys())

    plt.title(f"Latent space {reduction_type} of model activations {dist[0]} vs  {dist[1]}")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{dir}/latent_space_{reduction_type}_{dist[0]}_{dist[1]}_{lg1}.png")
    plt.show()

    return X_embedded

def compute_and_plot_reduction_change(act_refusal_comp, act_nn_refusal_comp, act_refusal_rep, act_nn_refusal_rep, dir, lg2, dist, reduction_type='pca', list_lg=[], n_compo=2, colors_type='lg'): 
    activations_ref_np_comp = act_refusal_comp.detach().cpu().to(torch.float16)
    activations_nn_ref_np_comp = act_nn_refusal_comp.detach().cpu().to(torch.float16)

    activations_ref_np_rep = act_refusal_rep.detach().cpu().to(torch.float16)
    activations_nn_ref_np_rep = act_nn_refusal_rep.detach().cpu().to(torch.float16)

    X_comp = torch.cat([activations_ref_np_comp, activations_nn_ref_np_comp], dim=0).numpy()
    X_rep = torch.cat([activations_ref_np_rep, activations_nn_ref_np_rep], dim=0).numpy()

    labels = [dist[0]] * activations_ref_np_rep.shape[0] + [dist[1]] * activations_nn_ref_np_rep.shape[0]

    if reduction_type=='pca': 
        pca = PCA(n_components=n_compo, random_state=42)
        X_embedded = pca.fit(X_comp).transform(X_rep)
    elif reduction_type=='tsne': 
        X_embedded = TSNE(n_components=n_compo, perplexity=30, random_state=42).fit(X_comp).transform(X_rep)
    elif reduction_type=='umap-unsupervised': 
        X_embedded = umap.UMAP(random_state=42).fit(X_comp).transform(X_rep)

    plt.figure(figsize=(10, 7)) 

    if list_lg==[]: 
        colors = {dist[0]: 'red', dist[1]: 'blue'}
        for i, (point, label) in enumerate(zip(X_embedded, labels)):
            plt.scatter(point[0], point[1], color=colors[label],  alpha=0.7, label=f"{label}")
        
        lang_handles = []
        for c in colors.values() :
            lang_handles.append(plt.scatter([], [], c=c, 
                                        marker='o', s=40, alpha=0.8, 
                                        edgecolors='black', linewidth=0.3))
        
        
        lang_legend = plt.legend(lang_handles, colors.keys(), 
                                title='Languages', loc='upper left', 
                                bbox_to_anchor=(1.02, 1), fontsize=9)
        
    else: 
        markers = {dist[0]: 'o', dist[1]: 's'}
        if colors_type=='cat': 
            colors = {'de':'lightcoral', 'bn':"maroon", 'ar':"peachpuff", 'jv':"deeppink", 'es':"khaki", 'mk':"deeppink", 'sw':"deeppink", 'tt':"deeppink", 'fr':"turquoise", 'ja':"teal", 'pt':"cyan", 'el':"steelblue", 'zh':"lavender", 'en':"navy", 'da':"blueviolet", 'lo':"deeppink", 'pag':"deeppink", 'mt':"deeppink"}
        elif colors_type=='lg':
            colors = {'de':'lightcoral', 'bn':"maroon", 'ar':"peachpuff", 'jv':"darkorange", 'es':"khaki", 'mk':"olive", 'sw':"yellow", 'tt':"lightgreen", 'fr':"turquoise", 'ja':"teal", 'pt':"cyan", 'el':"steelblue", 'zh':"lavender", 'en':"navy", 'da':"blueviolet", 'lo':"plum", 'pag':"fuchsia", 'mt':"deeppink"}
        for i, (point, label, lg) in enumerate(zip(X_embedded, labels, list_lg+list_lg)):
            plt.scatter(point[0], point[1], marker=markers[label], color=colors[lg], alpha=0.7)


        lang_handles = []
        for c in colors.values() :
            lang_handles.append(plt.scatter([], [], c=c, 
                                        marker='o', s=40, alpha=0.8, 
                                        edgecolors='black', linewidth=0.3))
        
        
        lang_legend = plt.legend(lang_handles, colors.keys(), 
                                title='Languages', loc='upper left', 
                                bbox_to_anchor=(1.02, 1), fontsize=9)
        
        plt.gca().add_artist(lang_legend)

        # Composition legend (markers)
        comp_handles = []
        for m in markers.values() :
            comp_handles.append(plt.scatter([], [], c='gray', 
                                        marker=m, 
                                        s=40, alpha=0.8, 
                                        edgecolors='black', linewidth=0.3))
        
        comp_legend = plt.legend(comp_handles, markers.keys(), 
                            title='Types', loc='upper left', 
                            bbox_to_anchor=(1.02, 0.3), fontsize=9)
    

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    #plt.legend(by_label.values(), by_label.keys())

    plt.title(f"Latent space {reduction_type} of model activations {dist[0]} vs  {dist[1]} in {lg2}")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{dir}/latent_space_{reduction_type}_{dist[0]}_{dist[1]}_{colors_type}_{lg2}.png")
    plt.show()

    return X_embedded

def create_background_classification(X_train_pca, train_labels, resolution=100):
    """
    Create a background classification based on nearest neighbors
    """
    # Get bounds of the plot
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    
    # Create grid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # For each grid point, find the nearest training sample
    train_refusal_mask = np.array(train_labels) == "Refusal"
    train_non_refusal_mask = np.array(train_labels) == "Non Refusal"
    
    refusal_points = X_train_pca[train_refusal_mask]
    non_refusal_points = X_train_pca[train_non_refusal_mask]
    
    # Calculate distances to each group
    dist_to_refusal = cdist(grid_points, refusal_points).min(axis=1)
    dist_to_non_refusal = cdist(grid_points, non_refusal_points).min(axis=1)
    
    # Classify each grid point based on nearest neighbor
    grid_classification = (dist_to_refusal < dist_to_non_refusal).astype(int)

    return xx, yy, grid_classification.reshape(xx.shape)


def compute_and_plot_reduction_with_refusal(activation_refusal_lg1, activation_nn_refusal_lg1, activation_refusal_lg2, activation_nn_refusal_lg2, activation_harmful_lg2, activation_harmless_lg2, dir, lg2, checkpoint, reduction_type='pca', list_lg=[], n_compo=2, colors_type='lg', prompt_type="vanilla"): 

    activations_ref_lg1_np = activation_refusal_lg1.detach().cpu().to(torch.float16)
    activations_nn_ref_lg1_np = activation_nn_refusal_lg1.detach().cpu().to(torch.float16)

    activations_ref_lg2_np = activation_refusal_lg2.detach().cpu().to(torch.float16)
    activations_nn_ref_lg2_np = activation_nn_refusal_lg2.detach().cpu().to(torch.float16)

    activations_harmful_np= activation_harmful_lg2.detach().cpu().to(torch.float16)
    activations_harmless_np = activation_harmless_lg2.detach().cpu().to(torch.float16)

    X_comp_lg1 = torch.cat([activations_ref_lg1_np, activations_nn_ref_lg1_np], dim=0).numpy()
    X_comp_lg2 = torch.cat([activations_ref_lg2_np, activations_nn_ref_lg2_np], dim=0).numpy()
    X_rep = torch.cat([activations_harmful_np, activations_harmless_np], dim=0).numpy()

    labels_rep = ["Harmful"] * activations_harmful_np.shape[0] + ["Harmless"] * activations_harmless_np.shape[0]
    labels_comp = ["Refusal"] * activations_ref_lg2_np.shape[0] + ["Non Refusal"] * activations_nn_ref_lg2_np.shape[0]
    labels_type = ["Vanilla"] * 150 + ["Adversarial"] * 150  + ["Vanilla"] * 150 + ["Adversarial"] * 150 

    if reduction_type=='pca': 
        pca = PCA(n_components=n_compo, random_state=42)
        X_comp_embedded_lg1 = pca.fit(X_comp_lg1)
        X_comp_embedded_lg2 = pca.transform(X_comp_lg2)
        X_rep_embedded = pca.transform(X_rep)
    elif reduction_type=='tsne': 
        tsne = TSNE(n_components=n_compo, perplexity=30, random_state=42)
        X_comp_embedded_lg1 = tsne.fit(X_comp_lg1)
        X_comp_embedded_lg2 = tsne.transform(X_comp_lg2)
        X_rep_embedded = tsne.transform(X_rep)
    elif reduction_type=='umap-unsupervised': 
        um = umap.UMAP(random_state=42)
        X_comp_embedded_lg1 = um.fit(X_comp_lg1)
        X_comp_embedded_lg2 = um.transform(X_comp_lg2)
        X_rep_embedded = um.transform(X_rep)

    plt.figure(figsize=(10, 7)) 


    # Create background classification
    xx, yy, grid_classification = create_background_classification(
        X_comp_embedded_lg2, labels_comp
    )
    
    # Plot background
    colors_bg = ['lightblue', 'lightcoral']  # Light colors for background
    plt.contourf(xx, yy, grid_classification, levels=1, colors=colors_bg, alpha=0.3)

    legend_elements_background = [
        Patch(facecolor='lightblue', edgecolor='k', label='Non Refusal'),
        Patch(facecolor='lightcoral', edgecolor='k', label='Refusal')
    ]

    legend_back = plt.legend(
        handles=legend_elements_background,
        title='Model decision',
        loc='upper left',            # Adjust as needed
        bbox_to_anchor=(1.02, 0.7),   # Prevent overlap with the other legend
        fontsize=9
    )
    plt.gca().add_artist(legend_back)

    if list_lg==[]: 
        colors = {"Harmful": 'red', "Harmless": 'blue'}
        if prompt_type=="all": 
            markers={"Vanilla":"o", "Adversarial":"s"}
            print("all")
            for i, (point, label, label_t) in enumerate(zip(X_rep_embedded, labels_rep, labels_type)):
                plt.scatter(point[0], point[1], color=colors[label], marker=markers[label_t], alpha=0.7, label=f"{label}-{label_t}")
            
            type_handles = []
            for c in markers.values() :
                type_handles.append(plt.scatter([], [], c="black", 
                                            marker=c, s=40, alpha=0.8, 
                                            edgecolors='black', linewidth=0.3))
            
            
            type_legend = plt.legend(type_handles, markers.keys(), 
                                    title='Type of prompts', loc='upper left', 
                                    bbox_to_anchor=(1.02, 0.85), fontsize=9)
        
            plt.gca().add_artist(type_legend)

        else : 
                    
            for i, (point, label) in enumerate(zip(X_rep_embedded, labels_rep)):
                plt.scatter(point[0], point[1], color=colors[label],  alpha=0.7, label=f"{label}")

        
        """
        colors = {"Refusal": 'red', "Non Refusal": 'blue'}

        for i, (point, label) in enumerate(zip(X_comp_embedded_lg2, labels_comp)):
            plt.scatter(point[0], point[1], color=colors[label], marker='x', alpha=1, label=f"{label}")
        
        lang_handles = []
        for c in colors.values() :
            lang_handles.append(plt.scatter([], [], c=c, 
                                        marker='x', s=40, alpha=1, 
                                        edgecolors='black', linewidth=0.3))
        
        
        lang_legend_comp = plt.legend(lang_handles, colors.keys(), 
                                title='Type of prompts', loc='upper left', 
                                bbox_to_anchor=(1.02, 0.85), fontsize=9)
        
        plt.gca().add_artist(lang_legend_comp)
        """
        
        lang_handles = []
        for c in colors.values() :
            lang_handles.append(plt.scatter([], [], c=c, 
                                        marker='o', s=40, alpha=0.8, 
                                        edgecolors='black', linewidth=0.3))
        
        
        lang_legend = plt.legend(lang_handles, colors.keys(), 
                                title='Type of prompts', loc='upper left', 
                                bbox_to_anchor=(1.02, 1), fontsize=9)
        
        #plt.gca().add_artist(lang_legend)
        
    else: 
        markers = {"Harmful": 'o',  "Harmless": 's'}
        if colors_type=='cat': 
            colors = {'de':'lightcoral', 'bn':"maroon", 'ar':"peachpuff", 'jv':"deeppink", 'es':"khaki", 'mk':"deeppink", 'sw':"deeppink", 'tt':"deeppink", 'fr':"turquoise", 'ja':"teal", 'pt':"cyan", 'el':"steelblue", 'zh':"lavender", 'en':"navy", 'da':"blueviolet", 'lo':"deeppink", 'pag':"deeppink", 'mt':"deeppink"}
        elif colors_type=='lg':
            colors = {'de':'lightcoral', 'bn':"maroon", 'ar':"peachpuff", 'jv':"darkorange", 'es':"khaki", 'mk':"olive", 'sw':"yellow", 'tt':"lightgreen", 'fr':"turquoise", 'ja':"teal", 'pt':"cyan", 'el':"steelblue", 'zh':"lavender", 'en':"navy", 'da':"blueviolet", 'lo':"plum", 'pag':"fuchsia", 'mt':"deeppink"}
        for i, (point, label, lg) in enumerate(zip(X_rep_embedded, labels_rep, list_lg+list_lg)):
            plt.scatter(point[0], point[1], marker=markers[label], color=colors[lg], alpha=0.7)


        lang_handles = []
        for c in colors.values() :
            lang_handles.append(plt.scatter([], [], c=c, 
                                        marker='o', s=40, alpha=0.8, 
                                        edgecolors='black', linewidth=0.3))
        
        
        lang_legend = plt.legend(lang_handles, colors.keys(), 
                                title='Languages', loc='upper left', 
                                bbox_to_anchor=(1.02, 1), fontsize=9)
        
        plt.gca().add_artist(lang_legend)

        # Composition legend (markers)
        comp_handles = []
        for m in markers.values() :
            comp_handles.append(plt.scatter([], [], c='gray', 
                                        marker=m, 
                                        s=40, alpha=0.8, 
                                        edgecolors='black', linewidth=0.3))
        
        comp_legend = plt.legend(comp_handles, markers.keys(), 
                            title='Types', loc='upper left', 
                            bbox_to_anchor=(1.02, 0.3), fontsize=9)
        
        #plt.gca().add_artist(comp_legend)
    

    #handles, labels = plt.gca().get_legend_handles_labels()

    #by_label = dict(zip(labels, handles))
    #plt.legend(by_label.values(), by_label.keys())

    plt.title(f"Latent space {reduction_type} of model activations Harmful vs Harmless with Refusal in {lg2} - {checkpoint}")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{dir}/latent_space_{reduction_type}_harmful_harmless_refusal_{colors_type}_{lg2}.png")
    plt.show()

    return X_rep_embedded


def compute_and_plot_reduction_with_classifier(activation_refusal_lg1, activation_nn_refusal_lg1, activation_harmful_lg2, activation_harmless_lg2, label_data_lg2_refusal,  dir, lg2, checkpoint, reduction_type='pca', list_lg=[], n_compo=2, colors_type='lg', prompt_type="vanilla", classifier="svm", classifier_form_prompts=False, layer=-1, lg_model="en"): 
    activations_ref_lg1_np = activation_refusal_lg1.detach().cpu().to(torch.float32)
    activations_nn_ref_lg1_np = activation_nn_refusal_lg1.detach().cpu().to(torch.float32)

    activations_harmful_np= activation_harmful_lg2.detach().cpu().to(torch.float32)
    activations_harmless_np = activation_harmless_lg2.detach().cpu().to(torch.float32)

    X_comp_lg1 = torch.cat([activations_ref_lg1_np, activations_nn_ref_lg1_np], dim=0).numpy()
    X_rep = torch.cat([activations_harmful_np, activations_harmless_np], dim=0).numpy()

    print(f"Shape: {X_comp_lg1.shape}")
    print(f"Total elements: {X_comp_lg1.size}")
    print(f"Inf values: {np.isinf(X_comp_lg1).sum()} ({100*np.isinf(X_comp_lg1).sum()/X_comp_lg1.size:.2f}%)")
    print(f"NaN values: {np.isnan(X_comp_lg1).sum()} ({100*np.isnan(X_comp_lg1).sum()/X_comp_lg1.size:.2f}%)")
    print(f"Finite values range: [{np.min(X_comp_lg1[np.isfinite(X_comp_lg1)])}, {np.max(X_comp_lg1[np.isfinite(X_comp_lg1)])}]")

    labels_rep = ["Harmful"] * activations_harmful_np.shape[0] + ["Harmless"] * activations_harmless_np.shape[0]
    labels_binary = [1] * activations_harmful_np.shape[0] + [0] * activations_harmless_np.shape[0]
    labels_type = ["Vanilla"] * 150 + ["Adversarial"] * 150  + ["Vanilla"] * 150 + ["Adversarial"] * 150 
    labels_refusal = ["Refusal" if x==1 else "Non refusal" for x in label_data_lg2_refusal]

    if reduction_type=='pca': 
        print("Compute PCA")
        pca = PCA(n_components=n_compo, random_state=42)
        X_comp_embedded_lg1 = pca.fit(X_comp_lg1)
        X_rep_embedded = pca.transform(X_rep)
    elif reduction_type=='tsne': 
        tsne = TSNE(n_components=n_compo, perplexity=30, random_state=42)
        X_comp_embedded_lg1 = tsne.fit(X_comp_lg1)
        X_rep_embedded = tsne.transform(X_rep)
    elif reduction_type=='umap-unsupervised': 
        um = umap.UMAP(random_state=42)
        X_comp_embedded_lg1 = um.fit(X_comp_lg1)
        X_rep_embedded = um.transform(X_rep)
    
    X_2d = X_rep_embedded[:, :2]

    # Scale the data for SVM
    scaler = StandardScaler()
    X_2d_scaled = scaler.fit_transform(X_2d)

    # Train linear 
    if classifier=="svm":
        classifier_model = SVC(kernel='linear', probability=True, random_state=42)
    elif classifier=="regression":
        classifier_model = LogisticRegression(random_state=42, max_iter=1000)
    print("Train classifier")
    classifier_model.fit(X_2d_scaled, labels_binary)

    # Create mesh grid for decision boundary
    h = 1  # step size in the mesh
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    # Scale the mesh grid points
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_scaled = scaler.transform(mesh_points)

    # Get decision function values for coloring background
    Z = classifier_model.decision_function(mesh_points_scaled)
    Z = Z.reshape(xx.shape)

    fig = plt.figure(figsize=(15, 6))
    ax_plot = fig.add_subplot(121)

    # Plot decision boundary background
    ax_plot.contourf(xx, yy, Z, levels=50, alpha=0.3, cmap='coolwarm')
    ax_plot.contour(xx, yy, Z, levels=[0], colors='black', linestyles='--', linewidths=2, label=f'{classifier} boundary harmlful vs harmless')

    if classifier_form_prompts==True:

        if classifier=="svm":
            classifier_model_form = SVC(kernel='linear', probability=True, random_state=42)
        elif classifier=="regression":
            classifier_model_form = LogisticRegression(random_state=42, max_iter=1000)
        classifier_model_form.fit(X_2d_scaled, labels_type)

        # Get decision function values for coloring background
        Z_form = classifier_model_form.decision_function(mesh_points_scaled)
        Z_form = Z_form.reshape(xx.shape)

        ax_plot.contour(xx, yy, Z_form, levels=[0], colors='black', linestyles='dotted', linewidths=2, label=f'{classifier} boundary adversarial vs vanilla')


        # Refusal vs nn refusal 
        if classifier=="svm":
            classifier_model_refusal = SVC(kernel='linear', probability=True, random_state=42)
        elif classifier=="regression":
            classifier_model_refusal = LogisticRegression(random_state=42, max_iter=1000)
        classifier_model_refusal.fit(X_2d_scaled, labels_refusal)

        # Get decision function values for coloring background
        Z_refusal= classifier_model_refusal.decision_function(mesh_points_scaled)
        Z_refusal = Z_refusal.reshape(xx.shape)

        ax_plot.contour(xx, yy, Z_refusal, levels=[0], colors='red', linestyles='dotted', linewidths=2, label=f'{classifier} boundary refusal vs non refusal')



    if list_lg==[]: 
        colors = {"Harmful": 'red', "Harmless": 'blue'}
        if prompt_type=="all": 
            markers={"Vanilla":"o", "Adversarial":"*"}
            edge_colors={"Refusal":"green", "Non refusal":None}
            print("all")
            for i, (point, label, label_t, label_ref) in enumerate(zip(X_rep_embedded, labels_rep, labels_type, labels_refusal )):
                ax_plot.scatter(point[0], point[1], color=colors[label], marker=markers[label_t], edgecolors=edge_colors[label_ref], alpha=0.7, label=f"{label}-{label_t}")
            
            type_handles = []
            for c in markers.values() :
                type_handles.append(plt.scatter([], [], c="black", 
                                            marker=c, s=40, alpha=0.8, 
                                            edgecolors='black', linewidth=0.3))
            
            
            #type_legend = plt.legend(type_handles, markers.keys(), 
            #                        title='Type of prompts', loc='upper left', 
            #                        bbox_to_anchor=(1.02, 0.85), fontsize=9)
        
                
            #plt.gca().add_artist(type_legend)

        else : 
                    
            for i, (point, label) in enumerate(zip(X_rep_embedded, labels_rep)):
                ax_plot.scatter(point[0], point[1], color=colors[label],  alpha=0.7, label=f"{label}")

        
        lang_handles = []
        for c in colors.values() :
            lang_handles.append(plt.scatter([], [], c=c, 
                                        marker='o', s=40, alpha=0.8, 
                                        edgecolors='black', linewidth=0.3))
        
        
        #lang_legend = plt.legend(lang_handles, colors.keys(), 
        #                        title='Type of prompts', loc='upper left', 
        #                        bbox_to_anchor=(1.02, 1), fontsize=9)
        # 
        #plt.gca().add_artist(lang_legend)
        
    else: 
        markers = {"Harmful": 'o',  "Harmless": 's'}
        if colors_type=='cat': 
            colors = {'de':'lightcoral', 'bn':"maroon", 'ar':"peachpuff", 'jv':"deeppink", 'es':"khaki", 'mk':"deeppink", 'sw':"deeppink", 'tt':"deeppink", 'fr':"turquoise", 'ja':"teal", 'pt':"cyan", 'el':"steelblue", 'zh':"lavender", 'en':"navy", 'da':"blueviolet", 'lo':"deeppink", 'pag':"deeppink", 'mt':"deeppink"}
        elif colors_type=='lg':
            colors = {'de':'lightcoral', 'bn':"maroon", 'ar':"peachpuff", 'jv':"darkorange", 'es':"khaki", 'mk':"olive", 'sw':"yellow", 'tt':"lightgreen", 'fr':"turquoise", 'ja':"teal", 'pt':"cyan", 'el':"steelblue", 'zh':"lavender", 'en':"navy", 'da':"blueviolet", 'lo':"plum", 'pag':"fuchsia", 'mt':"deeppink"}
        for i, (point, label, lg) in enumerate(zip(X_rep_embedded, labels_rep, list_lg+list_lg)):
            plt.scatter(point[0], point[1], marker=markers[label], color=colors[lg], alpha=0.7)


        lang_handles = []
        for c in colors.values() :
            lang_handles.append(plt.scatter([], [], c=c, 
                                        marker='o', s=40, alpha=0.8, 
                                        edgecolors='black', linewidth=0.3))
        
        
        lang_legend = plt.legend(lang_handles, colors.keys(), 
                                title='Languages', loc='upper left', 
                                bbox_to_anchor=(1.02, 1), fontsize=9)
        
        plt.gca().add_artist(lang_legend)

        # Composition legend (markers)
        comp_handles = []
        for m in markers.values() :
            comp_handles.append(plt.scatter([], [], c='gray', 
                                        marker=m, 
                                        s=40, alpha=0.8, 
                                        edgecolors='black', linewidth=0.3))
        
        comp_legend = plt.legend(comp_handles, markers.keys(), 
                            title='Types', loc='upper left', 
                            bbox_to_anchor=(1.02, 0.3), fontsize=9)
        
        #plt.gca().add_artist(comp_legend)
    

    #handles, labels = plt.gca().get_legend_handles_labels()

    #by_label = dict(zip(labels, handles))
    #plt.legend(by_label.values(), by_label.keys())

    #### Legends 
    clean_handles = []
    clean_labels = []

    # 1.colors
    for label, color in colors.items():
        clean_handles.append(plt.scatter([], [], c=color, marker='o', s=40, alpha=0.8, edgecolors='none'))
        clean_labels.append(label)

    # 2. markers
    for label, marker in markers.items():
        clean_handles.append(plt.scatter([], [], c="gray", marker=marker, s=40, alpha=0.8, edgecolors='none'))
        clean_labels.append(f"Type: {label}")

    # 3. edge
    for label, edge_color in edge_colors.items():
        clean_handles.append(plt.scatter([], [], c="white", marker='o', s=40, alpha=0.8, 
                                    edgecolors=edge_color, linewidth=2))
        clean_labels.append(f"Refusal: {label}")


    #######################
    ### Predictions all ###
    #######################
    results = {}
    results_all = {}
    results_vanilla = {}
    results_adversarial= {}


    # Harmful vs Harmless all 
    y_proba_all_harmfulness = classifier_model.predict_proba(X_2d_scaled)[:, 1]

    accuracy_all_harmfulness = classifier_model.score(X_2d_scaled, labels_binary)
    auc_score_all_harmfulness = roc_auc_score(labels_binary, y_proba_all_harmfulness)

    results_all["harmfulness"]={"accuracy":accuracy_all_harmfulness, 
                                "auc":auc_score_all_harmfulness}

    metrics_text_harmful_harmless = f"{classifier} Harmful vs Harmless:\nAccuracy: {accuracy_all_harmfulness:.3f}\nAUC: {auc_score_all_harmfulness:.3f}"

    # Vanilla vs Adversarial all
    y_proba_all_form = classifier_model_form.predict_proba(X_2d_scaled)[:, 1]

    accuracy_all_form = classifier_model_form.score(X_2d_scaled, labels_type)
    auc_score_all_form = roc_auc_score(labels_type, y_proba_all_form)

    results_all["adversarial_vs_vanilla"]={"accuracy":accuracy_all_form, 
                            "auc":auc_score_all_form}

    metrics_text_vanilla_adv = f"{classifier} Vanilla vs Adversarial:\nAccuracy: {accuracy_all_form:.3f}\nAUC: {auc_score_all_form:.3f}"

    # Refusal vs Non Refusal all 
    y_proba_all_refusal = classifier_model_refusal.predict_proba(X_2d_scaled)[:, 1]

    accuracy_all_refusal = classifier_model_refusal.score(X_2d_scaled, labels_refusal)
    auc_score_all_refusal = roc_auc_score(labels_refusal, y_proba_all_refusal)

    results_all["refusal"]={"accuracy":accuracy_all_refusal, 
                            "auc":auc_score_all_refusal}

    # Create metrics text for legend
    metrics_text_refusal = f"{classifier} Refusal vs Non refusal:\nAccuracy: {accuracy_all_refusal:.3f}\nAUC: {auc_score_all_refusal:.3f}"

    # Add metrics as a text entry in legend
    #legend_labels = list(by_label.keys()) + [metrics_text_harmful_harmless] + [metrics_text_vanilla_adv]+ [metrics_text_refusal]
    #legend_handles = list(by_label.values()) + [plt.Line2D([0], [0], color='black', alpha=0.7, linestyle='--')] + [plt.Line2D([0], [0], color='black', alpha=0.7, linestyle='dotted')] + [plt.Line2D([0], [0], color='red', alpha=0.7, linestyle='dotted')]

    legend_labels = clean_labels + [metrics_text_harmful_harmless, metrics_text_vanilla_adv, metrics_text_refusal]
    legend_handles = clean_handles + [
        plt.Line2D([0], [0], color='black', alpha=0.7, linestyle='--'),
        plt.Line2D([0], [0], color='black', alpha=0.7, linestyle='dotted'),
        plt.Line2D([0], [0], color='red', alpha=0.7, linestyle='dotted')
    ]


    ax_plot.legend(legend_handles, legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left')


    legend = ax_plot.legend(legend_handles, legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left')

    legend_bbox = legend.get_window_extent(fig.canvas.get_renderer())
    legend_bbox_fig = legend_bbox.transformed(fig.transFigure.inverted())

    table_left = legend_bbox_fig.x0
    table_bottom = legend_bbox_fig.y0 - 0.25  
    table_width = legend_bbox_fig.width
    table_height = 0.15

    ax_table = fig.add_axes([table_left, table_bottom, table_width, table_height])
    ax_table.axis('off')

    ax_plot.set_title(f"Latent space {reduction_type} of model activations Harmful vs Harmless with Refusal in {lg2} - {checkpoint}")
    ax_plot.set_xlabel("Dim 1")
    ax_plot.set_ylabel("Dim 2")
    ax_plot.grid(True)

    """

    file_path=f"gemma/{lg_model}"
    matrix = compute_refusal_matrix(file_path, checkpoint, lg2)

    row_labels = ['Vanilla', 'Adversarial']
    col_labels = ['Refusal', 'Over Refusal']

    table = ax_table.table(cellText=matrix,
                        rowLabels=row_labels,
                        colLabels=col_labels,
                        cellLoc='center',
                        loc='center')


    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    """
    if classifier_form_prompts==True: 
        name_fig = f"{dir}/latent_space_{reduction_type}_{lg2}_{classifier}_with_classification_layer_{layer}.png"
    else : 
        name_fig = f"{dir}/latent_space_{reduction_type}_{lg2}_{classifier}.png"
    plt.subplots_adjust(right=0.7)  
    plt.tight_layout()
    plt.savefig(name_fig)
    plt.show()

    #######################################
    ### Train classifier for each class ###
    #######################################

    print("Train classifiers for vanilla and adv")
    # prep data vanill & adv 

    print(isinstance(X_rep_embedded, np.ndarray))

    X_vanilla = np.concatenate([X_rep_embedded[:150], X_rep_embedded[300:450]], axis=0)
    X_adv = np.concatenate([X_rep_embedded[150:300], X_rep_embedded[450:]], axis=0)

    X_2d_vanilla = X_vanilla[:, :2]
    X_2d_adv = X_adv[:, :2]

    print(X_2d_vanilla.shape)
    print(X_2d_adv.shape)

    # Scale the data for SVM
    scaler = StandardScaler()
    X_2d_vanilla_scaled = scaler.fit_transform(X_2d_vanilla)
    X_2d_adv_scaled = scaler.fit_transform(X_2d_adv)

    print(label_data_lg2_refusal.tolist())
    refusal_label_data_list = label_data_lg2_refusal.tolist()
    print(refusal_label_data_list[:150]+refusal_label_data_list[300:450])
   
    labels_harmfulness = ["Harmful"] * 150 + ["Harmless"] * 150
    labels_refusal_vanilla = ["Refusal" if x==1 else "Non refusal" for x in refusal_label_data_list[:150]+refusal_label_data_list[300:450]]
    labels_refusal_adv = ["Refusal" if x==1 else "Non refusal" for x in refusal_label_data_list[150:300]+refusal_label_data_list[450:]]
    print("labels")
    print(labels_refusal_vanilla)
    print(labels_refusal_adv)

    if classifier=="svm":
        classifier_vanilla_harmfulness = SVC(kernel='linear', probability=True, random_state=42)
        classifier_vanilla_refusal = SVC(kernel='linear', probability=True, random_state=42)
        classifier_adv_harmfulness = SVC(kernel='linear', probability=True, random_state=42)
        classifier_adv_refusal = SVC(kernel='linear', probability=True, random_state=42)
    elif classifier=="regression":
        classifier_vanilla_harmfulness = LogisticRegression(random_state=42, max_iter=1000)
        classifier_vanilla_refusal = LogisticRegression(random_state=42, max_iter=1000)
        classifier_adv_harmfulness = LogisticRegression(random_state=42, max_iter=1000)
        classifier_adv_refusal = LogisticRegression(random_state=42, max_iter=1000)

    classifier_vanilla_harmfulness.fit(X_2d_vanilla_scaled, labels_harmfulness)
    classifier_vanilla_refusal.fit(X_2d_vanilla_scaled, labels_refusal_vanilla)
    classifier_adv_harmfulness.fit(X_2d_adv_scaled, labels_harmfulness)
    classifier_adv_refusal.fit(X_2d_adv_scaled, labels_refusal_adv)

    ##################################
    ### Predictions for each class ###
    ##################################

    # Harmful vs Harmless vanilla 
    y_proba_vanilla_harmfulness = classifier_vanilla_harmfulness.predict_proba(X_2d_vanilla_scaled)[:, 1]

    accuracy_vanilla_harmfulness= classifier_vanilla_harmfulness.score(X_2d_vanilla_scaled, labels_harmfulness)
    auc_score_vanilla_harmfulness = roc_auc_score(labels_harmfulness, y_proba_vanilla_harmfulness)

    # Harmful vs Harmless adversarial 
    y_proba_adv_harmfulness = classifier_adv_harmfulness.predict_proba(X_2d_adv_scaled)[:, 1]

    accuracy_adv_harmfulness = classifier_adv_harmfulness.score(X_2d_adv_scaled, labels_harmfulness)
    auc_score_adv_harmfulness = roc_auc_score(labels_harmfulness, y_proba_adv_harmfulness)

    # Refusal vs Non Refusal vanilla 
    y_proba_refusal_vanilla_refusal = classifier_vanilla_refusal.predict_proba(X_2d_vanilla_scaled)[:, 1]

    accuracy_vanilla_refusal = classifier_vanilla_refusal.score(X_2d_vanilla_scaled, labels_refusal_vanilla)
    auc_score_vanilla_refusal = roc_auc_score(labels_refusal_vanilla, y_proba_refusal_vanilla_refusal)

    # Refusal vs Non Refusal adversarial 
    y_proba_refusal_adv_refusal = classifier_adv_refusal.predict_proba(X_2d_adv_scaled)[:, 1]

    accuracy_adv_refusal = classifier_adv_refusal.score(X_2d_adv_scaled, labels_refusal_adv)
    auc_score_adv_refusal = roc_auc_score(labels_refusal_adv, y_proba_refusal_adv_refusal)

    results_vanilla["harmfulness"]={"accuracy":accuracy_vanilla_harmfulness, 
                            "auc":auc_score_vanilla_harmfulness}
    results_adversarial["harmfulness"]={"accuracy":accuracy_adv_harmfulness, 
                            "auc":auc_score_adv_harmfulness}
    results_vanilla["refusal"]={"accuracy":accuracy_vanilla_refusal, 
                            "auc":auc_score_vanilla_refusal}
    results_adversarial["refusal"]={"accuracy":accuracy_adv_refusal, 
                            "auc":auc_score_adv_refusal}
    
    results = {"all": results_all, 
               "vanilla": results_vanilla, 
               "adversarial": results_adversarial}
    
    path_json = f"{dir}/results_{reduction_type}_{lg2}_{classifier}_with_classification_layer_{layer}.json"
    with open(path_json, 'w') as f:
        json.dump(results, f, indent=4)

    return X_rep_embedded

def sum_dict(dict1, dict2):
    res = {}
    for key in dict1:
        res[key] = {}
        for second_key in dict1[key]:
            res[key][second_key] = (
                dict1[key][second_key] + 
                dict2[key][second_key]
            )
    return res

def compute_refusal_matrix(file_path, checkpoint, lg): 

    check = checkpoint.split("-")[-1]

    path_ref_json = f"{file_path}/result_refusal_adversarial_{check}.json"
    path_ove_json = f"{file_path}/result_over_refusal_adversarial_{check}.json"
    path_inc_json = f"{file_path}/result_incertain_harmful_adversarial_{check}.json"
    with open(path_ref_json) as json_file:
        data_ref = json.load(json_file)
    with open(path_ove_json) as json_file:
        data_over = json.load(json_file)
    with open(path_inc_json) as json_file:
        data_inc = json.load(json_file)

    data_r = sum_dict(data_ref, data_inc)

    data_refusal_adv = np.sum(list(data_r[lg].values()))
    data_over_refusal_adv = np.sum(list(data_over[lg].values()))

    path_ref_json = f"{file_path}/result_refusal_vanilla_{check}.json"
    path_ove_json = f"{file_path}/result_over_refusal_vanilla_{check}.json"
    path_inc_json = f"{file_path}/result_incertain_harmful_vanilla_{check}.json"
    with open(path_ref_json) as json_file:
        data_ref = json.load(json_file)
    with open(path_ove_json) as json_file:
        data_over = json.load(json_file)
    with open(path_inc_json) as json_file:
        data_inc = json.load(json_file)

    data_r = sum_dict(data_ref, data_inc)
    print(data_r)
    print(data_r[lg])
    print(data_r[lg].values())
    data_refusal_vanilla = np.sum(list(data_r[lg].values()))
    data_over_refusal_vanilla = np.sum(list(data_over[lg].values()))
     
    matrix = [[data_refusal_vanilla, data_over_refusal_vanilla], [data_refusal_adv, data_over_refusal_adv]]

    return matrix
