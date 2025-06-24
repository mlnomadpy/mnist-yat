import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean
from models import SingleLayerClassifier, PrototypeModel
from utils import cosine_similarity, euclidean_similarity, dot_product_similarity, yat_similarity

def evaluate(model, loader, device):
    """Evaluates any model on a given data loader."""
    model.to(device)
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            all_probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.concatenate(all_probs),
        'classification_report': classification_report(all_labels, all_preds, output_dict=True)
    }

def calculate_class_statistics(train_dataset):
    print("Calculating class statistics...")
    class_data = {i: [] for i in range(10)}
    for data, target in train_dataset:
        class_data[target].append(data.view(-1))

    class_stats = {}
    for i in range(10):
        data_stack = torch.stack(class_data[i])
        class_stats[i] = {
            'mean': data_stack.mean(0).numpy(),
            'std': data_stack.std(0).numpy(),
            'count': len(class_data[i])
        }
    return class_stats

def create_baseline_models(class_stats):
    """Creates all baseline models for comparison."""
    print("Creating enhanced baseline models...")
    prototypes = {i: class_stats[i]['mean'] for i in range(10)}

    baselines = {}
    baselines['random'] = SingleLayerClassifier()
    baselines['cosine_prototype'] = PrototypeModel(prototypes, cosine_similarity)
    baselines['euclidean_prototype'] = PrototypeModel(prototypes, euclidean_similarity)
    baselines['dot_product_prototype'] = PrototypeModel(prototypes, dot_product_similarity)
    baselines['yat_prototype'] = PrototypeModel(prototypes, yat_similarity)
    return baselines

def evaluate_all_models(trained_model, trained_yat_model, baselines, test_loader, device):
    """Evaluates both trained models and all baseline models."""
    print("Evaluating all models...")
    results = {}
    results['trained'] = evaluate(trained_model, test_loader, device)
    results['trained_yat'] = evaluate(trained_yat_model, test_loader, device)
    for name, model in baselines.items():
        print(f"  Evaluating {name} model...")
        results[name] = evaluate(model, test_loader, device)
    return results

def analyze_weights_and_similarities(trained_model, trained_yat_model, class_stats):
    """Calculates comprehensive similarity matrices between learned weights and class prototypes for both models."""
    print("Analyzing weight structure and similarities for both models...")
    
    results = {}
    # Analyze standard model weights
    weights = trained_model.linear.weight.data.cpu().numpy()
    results['weight_analysis'] = _analyze_model_weights(weights, class_stats)
    
    # Analyze Yat model weights
    yat_weights = trained_yat_model.linear.weight.data.cpu().numpy()
    results['weight_analysis_yat'] = _analyze_model_weights(yat_weights, class_stats)
    return results

def _analyze_model_weights(weights, class_stats):
    """Helper method to analyze weights for a specific model."""
    prototypes = {i: class_stats[i]['mean'] for i in range(10)}

    similarity_functions = {
        'Cosine': cosine_similarity,
        'Euclidean': euclidean_similarity,
        'Dot Product': dot_product_similarity,
        'Yat': yat_similarity
    }

    sim_matrices = {name: np.zeros((10, 10)) for name in similarity_functions}

    for i in range(10):  # Neuron
        for j in range(10):  # Class prototype
            w, p = weights[i], prototypes[j]
            for name, func in similarity_functions.items():
                sim_matrices[name][i, j] = func(w, p)

    # Additional analysis
    weight_prototype_distances = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            weight_prototype_distances[i, j] = euclidean(weights[i], prototypes[j])

    return {
        'similarity_matrices': sim_matrices,
        'weight_norms': np.linalg.norm(weights, axis=1),
        'prototype_norms': np.array([np.linalg.norm(prototypes[i]) for i in range(10)]),
        'weight_prototype_distances': weight_prototype_distances,
        'weights': weights,
        'prototypes': np.array([prototypes[i] for i in range(10)])
    }

def calculate_similarity_correlations(weight_analysis):
    """Calculates correlations between different similarity measures."""
    print("Calculating similarity correlations...")
    sim_matrices = weight_analysis['similarity_matrices']

    correlations = {}
    similarity_names = list(sim_matrices.keys());

    for i, name1 in enumerate(similarity_names):
        for j, name2 in enumerate(similarity_names):
            if i < j:  # Only calculate upper triangle
                matrix1 = sim_matrices[name1].flatten()
                matrix2 = sim_matrices[name2].flatten()
                corr, p_val = pearsonr(matrix1, matrix2)
                correlations[f"{name1} vs {name2}"] = {'correlation': corr, 'p_value': p_val}

    return correlations
