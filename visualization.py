import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage

def visualize_training_performance(results):
    """Visualizes training curves and model accuracy comparison for both models."""
    print("Generating training performance visualizations...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(24, 14), constrained_layout=True)
    fig.suptitle('Training and Performance Analysis - Standard vs YAT Models', fontsize=24, weight='bold')

    # Top row: Standard model
    _plot_training_curves_model(axes[0, 0], axes[0, 1], results['training_history'], 'Standard Model')
    _plot_performance_comparison_enhanced(axes[0, 2], results)
    
    # Bottom row: YAT model and comparison
    _plot_training_curves_model(axes[1, 0], axes[1, 1], results['yat_training_history'], 'YAT Model')
    _plot_models_comparison(axes[1, 2], results)

    plt.savefig('training_performance.png')
    plt.show()

def _plot_training_curves_model(ax_loss, ax_acc, history, title_prefix):
    """Plot training and validation curves for a specific model."""
    epochs = range(1, len(history['train_loss']) + 1)

    ax_loss.plot(epochs, history['train_loss'], 'o-', label='Train Loss', linewidth=2, markersize=4)
    ax_loss.plot(epochs, history['val_loss'], 's--', label='Val Loss', linewidth=2, markersize=4)
    ax_loss.set_title(f'{title_prefix} - Training & Validation Loss', fontsize=14, weight='bold')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(epochs, history['train_acc'], 'o-', label='Train Accuracy', linewidth=2, markersize=4)
    ax_acc.plot(epochs, history['val_acc'], 's--', label='Val Accuracy', linewidth=2, markersize=4)
    ax_acc.set_title(f'{title_prefix} - Training & Validation Accuracy', fontsize=14, weight='bold')
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

def _plot_performance_comparison_enhanced(ax, results):
    """Enhanced performance comparison with all models including both trained models."""
    model_names = [name for name in results if name not in ['training_history', 'yat_training_history', 'weight_analysis', 'weight_analysis_yat', 'similarity_correlations']]
    accuracies = [results[name]['accuracy'] for name in model_names]

    # Create color map with special colors for trained models
    colors = []
    for name in model_names:
        if name == 'trained':
            colors.append('darkblue')
        elif name == 'trained_yat':
            colors.append('darkred')
        else:
            colors.append(plt.cm.Set3(len(colors) / 10))

    bars = ax.bar(range(len(model_names)), accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax.set_title('Model Accuracy Comparison (All Models)', fontsize=14, weight='bold')
    ax.set_ylabel('Test Accuracy')
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels([name.replace('_', ' ').title() for name in model_names], rotation=45, ha='right')

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

    ax.grid(True, alpha=0.3, axis='y')

def _plot_models_comparison(ax, results):
    """Direct comparison between Standard and YAT models."""
    models = ['trained', 'trained_yat']
    model_labels = ['Standard Model', 'YAT Model']
    accuracies = [results[name]['accuracy'] for name in models]
    
    colors = ['darkblue', 'darkred']
    bars = ax.bar(model_labels, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    ax.set_title('Standard vs YAT Model Comparison', fontsize=14, weight='bold')
    ax.set_ylabel('Test Accuracy')
    ax.set_ylim(0, 1)
    
    # Add value labels and difference
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{acc:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Add difference annotation
    diff = accuracies[1] - accuracies[0]
    ax.text(0.5, max(accuracies) - 0.1, f'Difference: {diff:+.4f}', 
           ha='center', va='center', transform=ax.transData,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7),
           fontsize=11, fontweight='bold')
    
    ax.grid(True, alpha=0.3, axis='y')

def visualize_weights_and_prototypes(results):
    """Visualizes learned weights and class mean prototypes for both models."""
    print("Generating weight and prototype visualizations...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(24, 16), constrained_layout=True)
    fig.suptitle('Weight and Prototype Visualization - Standard vs YAT Models', fontsize=24, weight='bold')

    standard_weights = results['weight_analysis']['weights']
    yat_weights = results['weight_analysis_yat']['weights']
    prototypes = results['weight_analysis']['prototypes']

    # Determine a global min/max for consistent colormap scaling across weights and prototypes
    all_values = np.concatenate([standard_weights.flatten(), 
                                 yat_weights.flatten(), 
                                #  prototypes.flatten()
                                 ]
                                )
    vmin, vmax = all_values.min(), all_values.max()
    
    cmap = 'RdBu'

    # Standard model weights
    _plot_weight_templates_model(axes[0, 0], standard_weights, 'Standard Model Weights', cmap=cmap, vmin=vmin, vmax=vmax)
    
    # YAT model weights
    _plot_weight_templates_model(axes[0, 1], yat_weights, 'YAT Model Weights', cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Prototypes
    _plot_mean_prototypes(axes[0, 2], results, cmap=cmap)
    
    # Weight differences and comparisons
    _plot_weight_differences(axes[1, 0], standard_weights, yat_weights)
    _plot_weight_norms_comparison(axes[1, 1], results)
    _plot_weight_correlation_analysis(axes[1, 2], standard_weights, yat_weights)

    plt.savefig('weights_and_prototypes.png')
    plt.show()

def _create_image_grid(images, rows, cols):
    """Helper to create a grid of images."""
    # Assuming images are 28x28
    grid = images.reshape(rows, cols, 28, 28).swapaxes(1, 2).reshape(rows * 28, cols * 28)
    return grid

def _plot_weight_templates_model(ax, weights, title, cmap='viridis', vmin=None, vmax=None):
    """Enhanced weight templates visualization for a specific model."""
    grid = _create_image_grid(weights, rows=2, cols=5)
    im = ax.imshow(grid, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.axis('off')

    # Add colorbar
    fig = ax.get_figure()
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Weight Value')

    # Add class labels
    for i in range(10):
        row, col = i // 5, i % 5
        ax.text(col * 28 + 14, row * 28 + 30, f'Class {i}',
               ha='center', va='top', fontsize=10, weight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

def _plot_mean_prototypes(ax, results, cmap='viridis', vmin=None, vmax=None):
    """Enhanced mean prototypes visualization."""
    prototypes = results['weight_analysis']['prototypes']
    grid = _create_image_grid(prototypes, rows=2, cols=5)
    im = ax.imshow(grid, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.set_title('Class Mean Prototypes', fontsize=14, weight='bold')
    ax.axis('off')

    # Add colorbar
    fig = ax.get_figure()
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Pixel Intensity')

    # Add class labels
    for i in range(10):
        row, col = i // 5, i % 5
        ax.text(col * 28 + 14, row * 28 + 30, f'Class {i}',
               ha='center', va='top', fontsize=10, weight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

def _plot_weight_differences(ax, standard_weights, yat_weights):
    """Plot the differences between standard and YAT model weights."""
    weight_diff = standard_weights - yat_weights
    grid = _create_image_grid(weight_diff, rows=2, cols=5)
    
    vmax_abs = np.abs(weight_diff).max()
    im = ax.imshow(grid, cmap='RdBu_r', interpolation='nearest', vmin=-vmax_abs, vmax=vmax_abs)
    ax.set_title('Weight Differences (Standard - YAT)', fontsize=14, weight='bold')
    ax.axis('off')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Add class labels
    for i in range(10):
        row, col = i // 5, i % 5
        ax.text(col * 28 + 14, row * 28 + 30, f'Class {i}',
               ha='center', va='top', fontsize=10, weight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

def _plot_weight_norms_comparison(ax, results):
    """Compare weight norms between both models."""
    standard_norms = results['weight_analysis']['weight_norms']
    yat_norms = results['weight_analysis_yat']['weight_norms']
    
    x = np.arange(10)
    width = 0.35

    bars1 = ax.bar(x - width/2, standard_norms, width, label='Standard Model',
                  color='darkblue', edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x + width/2, yat_norms, width, label='YAT Model',
                  color='darkred', edgecolor='black', alpha=0.8)

    ax.set_title('Weight Norms Comparison', fontsize=14, weight='bold')
    ax.set_xlabel('Class Index')
    ax.set_ylabel('L2 Norm')
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.2f}', ha='center', va='bottom', fontsize=8)

def _plot_weight_correlation_analysis(ax, standard_weights, yat_weights):
    """Analyze correlation between standard and YAT model weights."""
    correlations = []
    for i in range(10):
        corr = np.corrcoef(standard_weights[i], yat_weights[i])[0, 1]
        correlations.append(corr)
    
    colors = ['green' if corr > 0.5 else 'orange' if corr > 0 else 'red' for corr in correlations]
    bars = ax.bar(range(10), correlations, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_title('Weight Correlation by Class\n(Standard vs YAT)', fontsize=14, weight='bold')
    ax.set_xlabel('Class Index')
    ax.set_ylabel('Correlation Coefficient')
    ax.set_ylim(-1, 1)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add correlation values
    for i, (bar, corr) in enumerate(zip(bars, correlations)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (0.05 if height >= 0 else -0.05),
               f'{corr:.3f}', ha='center', va='bottom' if height >= 0 else 'top',               fontweight='bold', fontsize=9)

def visualize_similarity_analysis(results):
    """Visualizes the heatmaps of different similarity measures for both models."""
    print("Generating similarity analysis heatmaps...")
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 12), constrained_layout=True)
    fig.suptitle('Similarity Analysis Comparison (Weights vs. Prototypes)', fontsize=24, weight='bold')

    # Standard model similarities
    standard_sim_matrices = results['weight_analysis']['similarity_matrices']
    sim_names = list(standard_sim_matrices.keys())
    
    for i, name in enumerate(sim_names):
        _plot_similarity_heatmap_model(axes[0, i], standard_sim_matrices[name], f'{name} - Standard Model')

    # YAT model similarities
    yat_sim_matrices = results['weight_analysis_yat']['similarity_matrices']
    
    for i, name in enumerate(sim_names):
        _plot_similarity_heatmap_model(axes[1, i], yat_sim_matrices[name], f'{name} - YAT Model')

    plt.savefig('similarity_analysis.png')
    plt.show()

def _plot_similarity_heatmap_model(ax, matrix, title):
    """Enhanced similarity heatmap for a specific model."""
    cmap_dict = {
        'Cosine - Standard Model': 'viridis', 'Cosine - YAT Model': 'plasma',
        'Euclidean - Standard Model': 'plasma', 'Euclidean - YAT Model': 'viridis',
        'Dot Product - Standard Model': 'magma', 'Dot Product - YAT Model': 'inferno',
        'Yat - Standard Model': 'inferno', 'Yat - YAT Model': 'magma'
    }

    sns.heatmap(matrix, annot=True, fmt='.2f', cmap=cmap_dict.get(title, 'viridis'),
               ax=ax, cbar_kws={'label': 'Similarity'}, square=True)
    ax.set_title(title, fontsize=12, weight='bold')
    ax.set_xlabel('Prototype Class')
    ax.set_ylabel('Weight Index')

def visualize_dimensionality_reduction(results):
    """Visualizes PCA, t-SNE, and clustering analysis."""
    print("Generating dimensionality reduction visualizations...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), constrained_layout=True)
    fig.suptitle('Dimensionality and Clustering Analysis', fontsize=24, weight='bold')

    weights = results['weight_analysis']['weights']
    prototypes = results['weight_analysis']['prototypes']

    _plot_pca_analysis(axes[0], weights, prototypes)
    _plot_tsne_analysis(axes[1], weights, prototypes)
    _plot_clustering_analysis(axes[2], weights, prototypes)

    plt.savefig('dimensionality_reduction.png')
    plt.show()

def _plot_pca_analysis(ax, weights, prototypes):
    """Enhanced PCA analysis."""
    pca = PCA(n_components=2)
    combined = np.vstack([weights, prototypes])
    transformed = pca.fit_transform(combined)

    # Plot prototypes
    ax.scatter(transformed[10:, 0], transformed[10:, 1], c='red', s=200,
              label='Prototypes', marker='X', edgecolors='black', linewidth=2)

    # Plot weights
    ax.scatter(transformed[:10, 0], transformed[:10, 1], c='blue', s=100,
              label='Weights', marker='o', edgecolors='black', linewidth=1)

    # Add labels
    for i in range(10):
        ax.annotate(str(i), transformed[i], xytext=(5, 5), textcoords='offset points',
                   fontsize=9, weight='bold')
        ax.annotate(str(i), transformed[i+10], xytext=(5, 5), textcoords='offset points',
                   fontsize=9, weight='bold', color='white')

    ax.set_title(f'PCA Analysis\n(Explained Var: {pca.explained_variance_ratio_.sum():.2f})',
                fontsize=14, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

def _plot_tsne_analysis(ax, weights, prototypes):
    """t-SNE visualization."""
    combined = np.vstack([weights, prototypes])
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    transformed = tsne.fit_transform(combined)

    # Plot prototypes
    ax.scatter(transformed[10:, 0], transformed[10:, 1], c='red', s=200,
              label='Prototypes', marker='X', edgecolors='black', linewidth=2)

    # Plot weights
    ax.scatter(transformed[:10, 0], transformed[:10, 1], c='blue', s=100,
              label='Weights', marker='o', edgecolors='black', linewidth=1)

    # Add labels
    for i in range(10):
        ax.annotate(str(i), transformed[i], xytext=(5, 5), textcoords='offset points',
                   fontsize=9, weight='bold')
        ax.annotate(str(i), transformed[i+10], xytext=(5, 5), textcoords='offset points',
                   fontsize=9, weight='bold', color='white')

    ax.set_title('t-SNE Analysis', fontsize=14, weight='bold')
    ax.legend()

def _plot_clustering_analysis(ax, weights, prototypes):
    """Hierarchical clustering analysis."""
    combined = np.vstack([weights, prototypes])
    labels = [f'W{i}' for i in range(10)] + [f'P{i}' for i in range(10)]
    linkage_matrix = linkage(combined, method='ward')

    dendrogram(linkage_matrix, labels=labels, ax=ax, leaf_rotation=90,
              leaf_font_size=10, color_threshold=0.7*max(linkage_matrix[:,2]))
    ax.set_title('Hierarchical Clustering\n(Ward Linkage)', fontsize=14, weight='bold')
    ax.set_ylabel('Distance')

def visualize_statistical_analysis(results, class_stats):
    """Visualizes various statistical comparisons."""
    print("Generating statistical analysis plots...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), constrained_layout=True)
    fig.suptitle('Statistical Analysis', fontsize=24, weight='bold')

    _plot_weight_norm_analysis(axes[0], results)
    _plot_similarity_correlations(axes[1], results)
    _plot_class_statistics(axes[2], class_stats)

    plt.savefig('statistical_analysis.png')
    plt.show()

def _plot_weight_norm_analysis(ax, results):
    """Enhanced weight norm analysis."""
    norms = results['weight_analysis']['weight_norms']
    proto_norms = results['weight_analysis']['prototype_norms']

    x = np.arange(10)
    width = 0.35

    bars1 = ax.bar(x - width/2, norms, width, label='Weight Norms',
                  color='skyblue', edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x + width/2, proto_norms, width, label='Prototype Norms',
                  color='lightcoral', edgecolor='black', alpha=0.8)

    ax.set_title('L2 Norm Comparison', fontsize=14, weight='bold')
    ax.set_xlabel('Class Index')
    ax.set_ylabel('L2 Norm')
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.2f}', ha='center', va='bottom', fontsize=8)

def _plot_similarity_correlations(ax, results):
    """Plot correlations between similarity measures."""
    correlations = results['similarity_correlations']

    pairs = list(correlations.keys())
    corr_values = [correlations[pair]['correlation'] for pair in pairs]
    colors = ['green' if corr > 0.5 else 'orange' if corr > 0 else 'red' for corr in corr_values]

    bars = ax.barh(range(len(pairs)), corr_values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_title('Similarity Measure Correlations', fontsize=14, weight='bold')
    ax.set_xlabel('Pearson Correlation')
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([pair.replace(' vs ', '\nvs\n') for pair in pairs], fontsize=10)
    ax.set_xlim(-1, 1)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3, axis='x')

    # Add correlation values
    for i, (bar, corr) in enumerate(zip(bars, corr_values)):
        ax.text(corr + (0.05 if corr >= 0 else -0.05), i, f'{corr:.3f}',
               va='center', ha='left' if corr >= 0 else 'right', fontweight='bold')

def _plot_class_statistics(ax, class_stats):
    """Plot class statistics."""
    counts = [class_stats[i]['count'] for i in range(10)]
    colors = plt.cm.tab10(np.arange(10))

    bars = ax.bar(range(10), counts, color=colors, alpha=0.8, edgecolor='black')
    ax.set_title('Training Samples per Class', fontsize=14, weight='bold')
    ax.set_xlabel('Class')
    ax.set_ylabel('Sample Count')
    ax.set_xticks(range(10))
    ax.grid(True, alpha=0.3, axis='y')

    # Add count labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 10,
               str(count), ha='center', va='bottom', fontweight='bold')

def visualize_detailed_performance(results):
    """Visualizes per-class performance and model comparison radar chart."""
    print("Generating detailed performance visualizations...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True)
    fig.suptitle('Detailed Performance Metrics', fontsize=24, weight='bold')

    _plot_per_class_performance(axes[0], results)
    _plot_model_comparison_radar(axes[1], results)

    plt.savefig('detailed_performance.png')
    plt.show()

def _plot_per_class_performance(ax, results):
    """Plot per-class performance metrics."""
    report = results['trained']['classification_report']

    classes = [str(i) for i in range(10)]
    precision = [report[cls]['precision'] for cls in classes]
    recall = [report[cls]['recall'] for cls in classes]
    f1_score = [report[cls]['f1-score'] for cls in classes]

    x = np.arange(10)
    width = 0.25

    bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8, color='skyblue')
    bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8, color='lightgreen')
    bars3 = ax.bar(x + width, f1_score, width, label='F1-Score', alpha=0.8, color='salmon')

    ax.set_title('Per-Class Performance Metrics', fontsize=14, weight='bold')
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

def _plot_model_comparison_radar(ax, results):
    """Radar chart comparing model accuracies."""
    model_names = ['trained', 'cosine_prototype', 'euclidean_prototype',
                  'dot_product_prototype', 'yat_prototype']
    model_labels = ['Trained', 'Cosine', 'Euclidean', 'Dot Product', 'Yat']

    if all(name in results for name in model_names):
        accuracies = [results[name]['accuracy'] for name in model_names]

        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(model_names), endpoint=False).tolist()
        accuracies += accuracies[:1]  # Complete the circle
        angles += angles[:1]

        ax.plot(angles, accuracies, 'o-', linewidth=2, color='blue', alpha=0.7)
        ax.fill(angles, accuracies, color='blue', alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(model_labels, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar', fontsize=14, weight='bold', pad=20)
        ax.grid(True)

        # Add accuracy values
        for angle, acc, label in zip(angles[:-1], accuracies[:-1], model_labels):
            ax.text(angle, acc + 0.05, f'{acc:.3f}', ha='center', va='center',
                   fontweight='bold', fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'Radar chart unavailable\n(missing model results)',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)

def visualize_error_analysis(results, test_dataset, test_loader):
    """Visualizes detailed error analysis and a gallery of misclassified images."""
    print("Generating error analysis visualizations...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True, gridspec_kw={'width_ratios': [1, 2]})
    fig.suptitle('Error Analysis', fontsize=24, weight='bold')

    _plot_error_analysis(axes[0], results)
    _plot_misclassified_gallery(axes[1], results, test_dataset, test_loader)

    plt.savefig('error_analysis.png')
    plt.show()

def _plot_error_analysis(ax, results):
    """Enhanced error analysis."""
    ax.set_title("Detailed Error Analysis", fontsize=16, weight='bold')
    ax.axis('off')

def _plot_misclassified_gallery(ax, results, test_dataset, test_loader):
    """Displays a gallery of misclassified images."""
    # Implementation for plotting misclassified images would go here
    pass
