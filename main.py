import torch
import warnings
from config import get_default_config
from data import load_data
from models import SingleLayerClassifier, SingleLayerYatClassifier
from training import train_model
from analysis import (
    calculate_class_statistics,
    create_baseline_models,
    evaluate_all_models,
    analyze_weights_and_similarities,
    calculate_similarity_correlations,
)
from visualization import (
    visualize_training_performance,
    visualize_weights_and_prototypes,
    visualize_similarity_analysis,
    visualize_dimensionality_reduction,
    visualize_statistical_analysis,
    visualize_detailed_performance,
    visualize_error_analysis,
)
from utils import set_seeds

def main():
    # Suppress warnings
    warnings.filterwarnings('ignore')

    # Configuration
    config = get_default_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seeds(config['seed'])

    # Load data
    train_loader, val_loader, test_loader, train_dataset, test_dataset = load_data(config)

    # Train models
    trained_model, standard_history = train_model(
        SingleLayerClassifier, 'standard', train_loader, val_loader, config, device
    )
    trained_yat_model, yat_history = train_model(
        SingleLayerYatClassifier, 'yat', train_loader, val_loader, config, device
    )

    # Analysis
    class_stats = calculate_class_statistics(train_dataset)
    baselines = create_baseline_models(class_stats)
    
    results = evaluate_all_models(trained_model, trained_yat_model, baselines, test_loader, device)
    results['training_history'] = standard_history
    results['yat_training_history'] = yat_history
    
    weight_analysis_results = analyze_weights_and_similarities(trained_model, trained_yat_model, class_stats)
    results.update(weight_analysis_results)

    results['similarity_correlations'] = calculate_similarity_correlations(results['weight_analysis'])

    # Visualization
    visualize_training_performance(results)
    visualize_weights_and_prototypes(results)
    visualize_similarity_analysis(results)
    visualize_dimensionality_reduction(results)
    visualize_statistical_analysis(results, class_stats)
    visualize_detailed_performance(results)
    visualize_error_analysis(results, test_dataset, test_loader)

if __name__ == '__main__':
    main()