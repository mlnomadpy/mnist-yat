import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine, euclidean
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import pearsonr
import warnings
from nmn.torch.nmn import YatNMN

warnings.filterwarnings('ignore')

class EnhancedSingleLayerExperiment:
    """
    An enhanced, comprehensive framework for training, evaluating, and analyzing a
    single-layer classifier on the MNIST dataset with extensive similarity comparisons.
    """

    def __init__(self, config=None):
        """Initializes the experiment with a given configuration."""
        self.config = self._get_default_config()
        if config:
            self.config.update(config)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._set_seeds(self.config['seed'])

        self.train_loader, self.val_loader, self.test_loader = None, None, None
        self.train_dataset, self.test_dataset = None, None
        self.class_stats = None
        self.trained_model = None
        self.baselines = {}
        self.results = {}

    def _get_default_config(self):
        """Returns the default configuration dictionary."""
        return {
            'seed': 42, 'batch_size': 128, 'validation_split': 0.1,
            'num_epochs': 25, 'lr': 0.01, 'weight_decay': 1e-4,
            'scheduler_step': 10, 'patience': 5, 'dropout': 0.1
        }

    def _set_seeds(self, seed):
        """Sets random seeds for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def _yat_similarity(v1, v2):
        """Calculates the Yat similarity: (dot_product^2) / (euclidean_distance^2)."""
        dot_product = np.dot(v1, v2)
        euclidean_dist_sq = np.sum((v1 - v2)**2)
        return (dot_product**2) / (euclidean_dist_sq + 1e-9)

    @staticmethod
    def _cosine_similarity(v1, v2):
        """Calculates cosine similarity."""
        return 1 - cosine(v1, v2)

    @staticmethod
    def _euclidean_similarity(v1, v2):
        """Calculates euclidean similarity (inverse of distance)."""
        return 1 / (1 + euclidean(v1, v2))

    @staticmethod
    def _dot_product_similarity(v1, v2):
        """Calculates dot product similarity."""
        return np.dot(v1, v2)

    class SingleLayerClassifier(nn.Module):
        """Standard single linear layer classifier."""
        def __init__(self, input_size=784, num_classes=10, dropout=0.0, bias=False):
            super().__init__()
            self.linear = nn.Linear(input_size, num_classes, bias=bias)
            self._initialize_weights()

        def _initialize_weights(self):
            nn.init.xavier_uniform_(self.linear.weight)
            if self.linear.bias is not None:
                nn.init.zeros_(self.linear.bias)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.linear(x)


    class SingleLayerYatClassifier(nn.Module):
        """Standard single linear layer classifier."""
        def __init__(self, input_size=784, num_classes=10, dropout=0.0, bias=False):
            super().__init__()
            self.linear = YatNMN(input_size, num_classes, bias=bias)
            self._initialize_weights()

        def _initialize_weights(self):
            nn.init.xavier_uniform_(self.linear.weight)
            if self.linear.bias is not None:
                nn.init.zeros_(self.linear.bias)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.linear(x)

    class PrototypeModel(nn.Module):
        """A generic prototype-based classifier."""
        def __init__(self, class_prototypes, similarity_fn):
            super().__init__()
            self.prototypes = class_prototypes
            self.similarity_fn = similarity_fn

        def forward(self, x):
            x_flat = x.view(x.size(0), -1).cpu().numpy()
            scores = np.zeros((x_flat.shape[0], len(self.prototypes)))
            for i, x_sample in enumerate(x_flat):
                for j, proto in self.prototypes.items():
                    scores[i, j] = self.similarity_fn(x_sample, proto)
            return torch.from_numpy(scores).float().to(x.device)

    def load_data(self):
        """Loads and preprocesses the MNIST dataset."""
        print("Loading MNIST dataset...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        full_train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        self.test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        num_train = len(full_train_dataset)
        val_size = int(self.config['validation_split'] * num_train)
        train_size = num_train - val_size

        self.train_dataset, val_dataset = torch.utils.data.random_split(
            full_train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config['seed'])
        )

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1000, shuffle=False)    
    def train_model(self, model_class, model_name, verbose=True):
        """Generic method to train any model class."""
        print(f"Training {model_name} on {self.device}...")
        model = model_class(dropout=self.config['dropout']).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(), lr=self.config['lr'],
            momentum=0.9, weight_decay=self.config['weight_decay']
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.config['scheduler_step'], gamma=0.1)

        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        best_val_loss = float('inf')
        patience_counter = 0

        if verbose:
            print(f"Training {model_name}:")
            print("Epoch | Train Loss | Val Loss | Train Acc | Val Acc | LR")
            print("-" * 65)

        for epoch in range(self.config['num_epochs']):
            # Training phase
            model.train()
            train_loss, train_correct, train_total = 0, 0, 0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                train_correct += (outputs.argmax(1) == labels).sum().item()
                train_total += labels.size(0)

            # Validation phase
            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    val_correct += (outputs.argmax(1) == labels).sum().item()
                    val_total += labels.size(0)

            history['train_loss'].append(train_loss / train_total)
            history['val_loss'].append(val_loss / val_total)
            history['train_acc'].append(train_correct / train_total)
            history['val_acc'].append(val_correct / val_total)
            current_lr = optimizer.param_groups[0]['lr']

            if verbose:
                print(f"{epoch+1:5d} | {history['train_loss'][-1]:10.4f} | "
                      f"{history['val_loss'][-1]:8.4f} | {history['train_acc'][-1]:9.4f} | "
                      f"{history['val_acc'][-1]:7.4f} | {current_lr:.6f}")
            scheduler.step()

            if history['val_loss'][-1] < best_val_loss:
                best_val_loss = history['val_loss'][-1]
                patience_counter = 0
                torch.save(model.state_dict(), f'best_{model_name}_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    if verbose: print(f"Early stopping at epoch {epoch+1}")
                    break

        model.load_state_dict(torch.load(f'best_{model_name}_model.pth'))
        return model, history

    def train(self, verbose=True):
        """Trains both SingleLayerClassifier and SingleLayerYatClassifier."""
        print(f"Training both models on {self.device}...")
        
        # Train standard classifier
        self.trained_model, standard_history = self.train_model(
            self.SingleLayerClassifier, 'standard', verbose
        )
        
        # Train Yat classifier
        self.trained_yat_model, yat_history = self.train_model(
            self.SingleLayerYatClassifier, 'yat', verbose
        )
        
        self.results['training_history'] = standard_history
        self.results['yat_training_history'] = yat_history

    def evaluate(self, model, loader):
        """Evaluates any model on a given data loader."""
        model.to(self.device)
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
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

    def analyze(self):
        """Performs a comprehensive analysis of the trained model and baselines."""
        print("\n--- Starting Enhanced Analysis ---")
        self._calculate_class_statistics()
        self._create_baseline_models()
        self._evaluate_all_models()
        self._analyze_weights_and_similarities()
        self._calculate_similarity_correlations()
        print("--- Analysis Complete ---")

    def _calculate_class_statistics(self):
        print("Calculating class statistics...")
        class_data = {i: [] for i in range(10)}
        for data, target in self.train_dataset:
            class_data[target].append(data.view(-1))

        self.class_stats = {}
        for i in range(10):
            data_stack = torch.stack(class_data[i])
            self.class_stats[i] = {
                'mean': data_stack.mean(0).numpy(),
                'std': data_stack.std(0).numpy(),
                'count': len(class_data[i])
            }

    def _create_baseline_models(self):
        """Creates all baseline models for comparison."""
        print("Creating enhanced baseline models...")
        prototypes = {i: self.class_stats[i]['mean'] for i in range(10)}

        # All similarity-based models
        self.baselines['random'] = self.SingleLayerClassifier()
        self.baselines['cosine_prototype'] = self.PrototypeModel(prototypes, self._cosine_similarity)
        self.baselines['euclidean_prototype'] = self.PrototypeModel(prototypes, self._euclidean_similarity)
        self.baselines['dot_product_prototype'] = self.PrototypeModel(prototypes, self._dot_product_similarity)
        self.baselines['yat_prototype'] = self.PrototypeModel(prototypes, self._yat_similarity)    
    def _evaluate_all_models(self):
        """Evaluates both trained models and all baseline models."""
        print("Evaluating all models...")
        self.results['trained'] = self.evaluate(self.trained_model, self.test_loader)
        self.results['trained_yat'] = self.evaluate(self.trained_yat_model, self.test_loader)
        for name, model in self.baselines.items():
            print(f"  Evaluating {name} model...")
            self.results[name] = self.evaluate(model, self.test_loader)    
    def _analyze_weights_and_similarities(self):
        """Calculates comprehensive similarity matrices between learned weights and class prototypes for both models."""
        print("Analyzing weight structure and similarities for both models...")
        
        # Analyze standard model weights
        weights = self.trained_model.linear.weight.data.cpu().numpy()
        self._analyze_model_weights(weights, 'standard')
        
        # Analyze Yat model weights
        yat_weights = self.trained_yat_model.linear.weight.data.cpu().numpy()
        self._analyze_model_weights(yat_weights, 'yat')

    def _analyze_model_weights(self, weights, model_name):
        """Helper method to analyze weights for a specific model."""
        prototypes = {i: self.class_stats[i]['mean'] for i in range(10)}

        similarity_functions = {
            'Cosine': self._cosine_similarity,
            'Euclidean': self._euclidean_similarity,
            'Dot Product': self._dot_product_similarity,
            'Yat': self._yat_similarity
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

        weight_analysis_key = f'weight_analysis_{model_name}' if model_name != 'standard' else 'weight_analysis'
        self.results[weight_analysis_key] = {
            'similarity_matrices': sim_matrices,
            'weight_norms': np.linalg.norm(weights, axis=1),
            'prototype_norms': np.array([np.linalg.norm(prototypes[i]) for i in range(10)]),
            'weight_prototype_distances': weight_prototype_distances,
            'weights': weights,
            'prototypes': np.array([prototypes[i] for i in range(10)])
        }

    def _calculate_similarity_correlations(self):
        """Calculates correlations between different similarity measures."""
        print("Calculating similarity correlations...")
        sim_matrices = self.results['weight_analysis']['similarity_matrices']

        correlations = {}
        similarity_names = list(sim_matrices.keys());

        for i, name1 in enumerate(similarity_names):
            for j, name2 in enumerate(similarity_names):
                if i < j:  # Only calculate upper triangle
                    matrix1 = sim_matrices[name1].flatten()
                    matrix2 = sim_matrices[name2].flatten()
                    corr, p_val = pearsonr(matrix1, matrix2)
                    correlations[f"{name1} vs {name2}"] = {'correlation': corr, 'p_value': p_val}

        self.results['similarity_correlations'] = correlations    
    def visualize_training_performance(self):
        """Visualizes training curves and model accuracy comparison for both models."""
        print("Generating training performance visualizations...")
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(2, 3, figsize=(24, 14), constrained_layout=True)
        fig.suptitle('Training and Performance Analysis - Standard vs YAT Models', fontsize=24, weight='bold')

        # Top row: Standard model
        self._plot_training_curves_model(axes[0, 0], axes[0, 1], 'training_history', 'Standard Model')
        self._plot_performance_comparison_enhanced(axes[0, 2])
        
        # Bottom row: YAT model and comparison
        self._plot_training_curves_model(axes[1, 0], axes[1, 1], 'yat_training_history', 'YAT Model')
        self._plot_models_comparison(axes[1, 2])

        plt.show()

    def _plot_training_curves_model(self, ax_loss, ax_acc, history_key, title_prefix):
        """Plot training and validation curves for a specific model."""
        history = self.results[history_key]
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

    def _plot_performance_comparison_enhanced(self, ax):
        """Enhanced performance comparison with all models including both trained models."""
        model_names = [name for name in self.results if name not in ['training_history', 'yat_training_history', 'weight_analysis', 'weight_analysis_yat', 'similarity_correlations']]
        accuracies = [self.results[name]['accuracy'] for name in model_names]

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

    def _plot_models_comparison(self, ax):
        """Direct comparison between Standard and YAT models."""
        models = ['trained', 'trained_yat']
        model_labels = ['Standard Model', 'YAT Model']
        accuracies = [self.results[name]['accuracy'] for name in models]
        
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
    def visualize_weights_and_prototypes(self):
        """Visualizes learned weights and class mean prototypes for both models."""
        print("Generating weight and prototype visualizations...")
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(2, 3, figsize=(24, 16), constrained_layout=True)
        fig.suptitle('Weight and Prototype Visualization - Standard vs YAT Models', fontsize=24, weight='bold')

        # Standard model weights
        standard_weights = self.results['weight_analysis']['weights']
        self._plot_weight_templates_model(axes[0, 0], standard_weights, 'Standard Model Weights')
        
        # YAT model weights
        yat_weights = self.results['weight_analysis_yat']['weights']
        self._plot_weight_templates_model(axes[0, 1], yat_weights, 'YAT Model Weights')
        
        # Prototypes
        self._plot_mean_prototypes(axes[0, 2])
        
        # Weight differences and comparisons
        self._plot_weight_differences(axes[1, 0], standard_weights, yat_weights)
        self._plot_weight_norms_comparison(axes[1, 1])
        self._plot_weight_correlation_analysis(axes[1, 2], standard_weights, yat_weights)

        plt.show()

    def _plot_weight_templates_model(self, ax, weights, title):
        """Enhanced weight templates visualization for a specific model."""
        grid = self._create_image_grid(weights, rows=2, cols=5)
        im = ax.imshow(grid, cmap='RdBu_r', interpolation='nearest')
        ax.set_title(title, fontsize=14, weight='bold')
        ax.axis('off')

        # Add class labels
        for i in range(10):
            row, col = i // 5, i % 5
            ax.text(col * 28 + 14, row * 28 + 30, f'Class {i}',
                   ha='center', va='top', fontsize=10, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    def _plot_weight_differences(self, ax, standard_weights, yat_weights):
        """Plot the differences between standard and YAT model weights."""
        weight_diff = standard_weights - yat_weights
        grid = self._create_image_grid(weight_diff, rows=2, cols=5)
        im = ax.imshow(grid, cmap='RdBu_r', interpolation='nearest')
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

    def _plot_weight_norms_comparison(self, ax):
        """Compare weight norms between both models."""
        standard_norms = self.results['weight_analysis']['weight_norms']
        yat_norms = self.results['weight_analysis_yat']['weight_norms']
        
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

    def _plot_weight_correlation_analysis(self, ax, standard_weights, yat_weights):
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
                   f'{corr:.3f}', ha='center', va='bottom' if height >= 0 else 'top',                   fontweight='bold', fontsize=9)

    def visualize_similarity_analysis(self):
        """Visualizes the heatmaps of different similarity measures for both models."""
        print("Generating similarity analysis heatmaps...")
        plt.style.use('seaborn-v0_8-whitegrid')
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 12), constrained_layout=True)
        fig.suptitle('Similarity Analysis Comparison (Weights vs. Prototypes)', fontsize=24, weight='bold')

        # Standard model similarities
        standard_sim_matrices = self.results['weight_analysis']['similarity_matrices']
        sim_names = list(standard_sim_matrices.keys())
        
        for i, name in enumerate(sim_names):
            self._plot_similarity_heatmap_model(axes[0, i], standard_sim_matrices[name], f'{name} - Standard Model')

        # YAT model similarities
        yat_sim_matrices = self.results['weight_analysis_yat']['similarity_matrices']
        
        for i, name in enumerate(sim_names):
            self._plot_similarity_heatmap_model(axes[1, i], yat_sim_matrices[name], f'{name} - YAT Model')

        plt.show()

    def _plot_similarity_heatmap_model(self, ax, matrix, title):
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

    def visualize_dimensionality_reduction(self):
        """Visualizes PCA, t-SNE, and clustering analysis."""
        print("Generating dimensionality reduction visualizations...")
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(1, 3, figsize=(24, 8), constrained_layout=True)
        fig.suptitle('Dimensionality and Clustering Analysis', fontsize=24, weight='bold')

        weights = self.results['weight_analysis']['weights']
        prototypes = self.results['weight_analysis']['prototypes']

        self._plot_pca_analysis(axes[0], weights, prototypes)
        self._plot_tsne_analysis(axes[1], weights, prototypes)
        self._plot_clustering_analysis(axes[2], weights, prototypes)

        plt.show()

    def visualize_statistical_analysis(self):
        """Visualizes various statistical comparisons."""
        print("Generating statistical analysis plots...")
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(1, 3, figsize=(24, 7), constrained_layout=True)
        fig.suptitle('Statistical Analysis', fontsize=24, weight='bold')

        self._plot_weight_norm_analysis(axes[0])
        self._plot_similarity_correlations(axes[1])
        self._plot_class_statistics(axes[2])

        plt.show()

    def visualize_detailed_performance(self):
        """Visualizes per-class performance and model comparison radar chart."""
        print("Generating detailed performance visualizations...")
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True)
        fig.suptitle('Detailed Performance Metrics', fontsize=24, weight='bold')

        self._plot_per_class_performance(axes[0])
        self._plot_model_comparison_radar(axes[1])

        plt.show()

    def visualize_error_analysis(self):
        """Visualizes detailed error analysis and a gallery of misclassified images."""
        print("Generating error analysis visualizations...")
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True, gridspec_kw={'width_ratios': [1, 2]})
        fig.suptitle('Error Analysis', fontsize=24, weight='bold')

        self._plot_error_analysis(axes[0])
        self._plot_misclassified_gallery(axes[1])

        plt.show()

    def _plot_training_curves(self, ax_loss, ax_acc):
        """Plot training and validation curves."""
        history = self.results['training_history']
        epochs = range(1, len(history['train_loss']) + 1)

        ax_loss.plot(epochs, history['train_loss'], 'o-', label='Train Loss', linewidth=2, markersize=4)
        ax_loss.plot(epochs, history['val_loss'], 's--', label='Val Loss', linewidth=2, markersize=4)
        ax_loss.set_title('Training & Validation Loss', fontsize=14, weight='bold')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)

        ax_acc.plot(epochs, history['train_acc'], 'o-', label='Train Accuracy', linewidth=2, markersize=4)
        ax_acc.plot(epochs, history['val_acc'], 's--', label='Val Accuracy', linewidth=2, markersize=4)
        ax_acc.set_title('Training & Validation Accuracy', fontsize=14, weight='bold')
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)

    def _plot_performance_comparison(self, ax):
        """Enhanced performance comparison with error bars."""
        model_names = [name for name in self.results if name not in ['training_history', 'weight_analysis', 'similarity_correlations']]
        accuracies = [self.results[name]['accuracy'] for name in model_names]

        # Create color map
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))

        bars = ax.bar(range(len(model_names)), accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax.set_title('Model Accuracy Comparison', fontsize=14, weight='bold')
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

    def _plot_weight_templates(self, ax):
        """Enhanced weight templates visualization."""
        weights = self.results['weight_analysis']['weights']
        grid = self._create_image_grid(weights, rows=2, cols=5)
        im = ax.imshow(grid, cmap='RdBu_r', interpolation='nearest')
        ax.set_title('Learned Weight Templates', fontsize=14, weight='bold')
        ax.axis('off')

        # Add class labels
        for i in range(10):
            row, col = i // 5, i % 5
            ax.text(col * 28 + 14, row * 28 + 30, f'Class {i}',
                   ha='center', va='top', fontsize=10, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    def _plot_mean_prototypes(self, ax):
        """Enhanced mean prototypes visualization."""
        prototypes = self.results['weight_analysis']['prototypes']
        grid = self._create_image_grid(prototypes, rows=2, cols=5)
        im = ax.imshow(grid, cmap='viridis', interpolation='nearest')
        ax.set_title('Class Mean Prototypes', fontsize=14, weight='bold')
        ax.axis('off')

        # Add class labels
        for i in range(10):
            row, col = i // 5, i % 5
            ax.text(col * 28 + 14, row * 28 + 30, f'Class {i}',
                   ha='center', va='top', fontsize=10, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    def _plot_confusion_matrix_unified(self, ax, model_name, title_suffix=""):
        if model_name not in self.results:
            ax.axis('off')
            ax.text(0.5, 0.5, f'{model_name} not available',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return

        cm = self.results[model_name]['confusion_matrix']
        acc = self.results[model_name]['accuracy']

        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        # Create combined annotations (count + percentage)
        annotations = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                percent = cm_percent[i, j]
                if count == 0:
                    annotations[i, j] = "0\n(0%)"
                else:
                    annotations[i, j] = f"{count}\n({percent:.1f}%)"

        # Color map selection based on model type
        if 'prototype' in model_name:
            cmap = 'Greens'
        elif model_name == 'trained':
            cmap = 'Blues'
        elif model_name == 'random':
            cmap = 'Oranges'
        else:
            cmap = 'Purples'

        # Create heatmap
        sns.heatmap(cm, annot=annotations, fmt='', cmap=cmap, ax=ax,
                   cbar_kws={'label': 'Count'}, square=True,
                   annot_kws={'fontsize': 8, 'fontweight': 'bold'})

        # Format title
        display_name = model_name.replace('_', ' ').title() + title_suffix
        ax.set_title(f'{display_name}\n(Accuracy: {acc:.3f})',
                    fontsize=12, weight='bold')
        ax.set_xlabel('Predicted Class', fontsize=10)
        ax.set_ylabel('True Class', fontsize=10)

        # Improve tick labels
        ax.set_xticklabels(range(10), fontsize=9)
        ax.set_yticklabels(range(10), fontsize=9, rotation=0)

    def _plot_trained_confusion_matrix(self, ax):
        """Enhanced confusion matrix for trained model."""
        self._plot_confusion_matrix_unified(ax, 'trained', ' Model')

    def _plot_similarity_heatmap(self, ax, matrix, name):
        """Enhanced similarity heatmap."""
        cmap_dict = {
            'Cosine': 'viridis',
            'Euclidean': 'plasma',
            'Dot Product': 'magma',
            'Yat': 'inferno'
        }

        sns.heatmap(matrix, annot=True, fmt='.2f', cmap=cmap_dict.get(name, 'viridis'),
                   ax=ax, cbar_kws={'label': 'Similarity'}, square=True)
        ax.set_title(f'{name} Similarity\n(Weights vs Prototypes)', fontsize=14, weight='bold')
        ax.set_xlabel('Prototype Class')
        ax.set_ylabel('Weight Index')
    def _plot_baseline_confusion_matrix(self, ax, name):
        """Enhanced baseline confusion matrix using unified method."""
        self._plot_confusion_matrix_unified(ax, name)

    def _plot_pca_analysis(self, ax, weights, prototypes):
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

    def _plot_tsne_analysis(self, ax, weights, prototypes):
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

    def _plot_clustering_analysis(self, ax, weights, prototypes):
        """Hierarchical clustering analysis."""
        combined = np.vstack([weights, prototypes])
        labels = [f'W{i}' for i in range(10)] + [f'P{i}' for i in range(10)]
        linkage_matrix = linkage(combined, method='ward')

        dendrogram(linkage_matrix, labels=labels, ax=ax, leaf_rotation=90,
                  leaf_font_size=10, color_threshold=0.7*max(linkage_matrix[:,2]))
        ax.set_title('Hierarchical Clustering\n(Ward Linkage)', fontsize=14, weight='bold')
        ax.set_ylabel('Distance')

    def _plot_weight_norm_analysis(self, ax):
        """Enhanced weight norm analysis."""
        norms = self.results['weight_analysis']['weight_norms']
        proto_norms = self.results['weight_analysis']['prototype_norms']

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

    def _plot_similarity_correlations(self, ax):
        """Plot correlations between similarity measures."""
        correlations = self.results['similarity_correlations']

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

    def _plot_class_statistics(self, ax):
        """Plot class statistics."""
        counts = [self.class_stats[i]['count'] for i in range(10)]
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

    def _plot_per_class_performance(self, ax):
        """Plot per-class performance metrics."""
        report = self.results['trained']['classification_report']

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

    def _plot_model_comparison_radar(self, ax):
        """Radar chart comparing model accuracies."""
        model_names = ['trained', 'cosine_prototype', 'euclidean_prototype',
                      'dot_product_prototype', 'yat_prototype']
        model_labels = ['Trained', 'Cosine', 'Euclidean', 'Dot Product', 'Yat']

        if all(name in self.results for name in model_names):
            accuracies = [self.results[name]['accuracy'] for name in model_names]

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

    def _plot_error_analysis(self, ax):
        """Enhanced error analysis."""
        ax.set_title("Detailed Error Analysis", fontsize=16, weight='bold')
        ax.axis('off')

        res = self.results['trained']
        correct_mask = res['predictions'] == res['labels']
        confidences = res['probabilities'].max(axis=1)

        # Calculate statistics
        correct_conf_mean = confidences[correct_mask].mean()
        correct_conf_std = confidences[correct_mask].std()
        incorrect_conf_mean = confidences[~correct_mask].mean()
        incorrect_conf_std = confidences[~correct_mask].std()

        # Find most common mistakes
        common_mistakes = {}
        for true, pred in zip(res['labels'][~correct_mask], res['predictions'][~correct_mask]):
            pair = (true, pred)
            common_mistakes[pair] = common_mistakes.get(pair, 0) + 1

        top_mistakes = sorted(common_mistakes.items(), key=lambda x: x[1], reverse=True)[:6]

        # Calculate per-class error rates
        class_errors = {}
        for i in range(10):
            class_mask = res['labels'] == i
            if class_mask.sum() > 0:
                error_rate = 1 - (res['predictions'][class_mask] == i).mean()
                class_errors[i] = error_rate

        worst_classes = sorted(class_errors.items(), key=lambda x: x[1], reverse=True)[:3]

        text = f"""CONFIDENCE ANALYSIS:
• Correct Predictions: {correct_conf_mean:.3f} ± {correct_conf_std:.3f}
• Incorrect Predictions: {incorrect_conf_mean:.3f} ± {incorrect_conf_std:.3f}
• Confidence Gap: {correct_conf_mean - incorrect_conf_mean:.3f}

MOST COMMON ERRORS:
{chr(10).join([f'• {true} → {pred}: {count} times ({count/len(res["labels"])*100:.1f}%)' for (true, pred), count in top_mistakes])}

HIGHEST ERROR RATES:
{chr(10).join([f'• Class {cls}: {rate*100:.1f}% error rate' for cls, rate in worst_classes])}

OVERALL STATISTICS:
• Total Errors: {(~correct_mask).sum()} / {len(res['labels'])}
• Error Rate: {(~correct_mask).mean()*100:.2f}%
• Accuracy: {correct_mask.mean()*100:.2f}%"""

        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    def _plot_misclassified_gallery(self, ax):
        """Enhanced misclassified images gallery."""
        ax.set_title("Gallery of Misclassified Examples", fontsize=16, weight='bold')
        ax.axis('off')

        res = self.results['trained']
        misclassified_indices = np.where(res['predictions'] != res['labels'])[0]

        if len(misclassified_indices) == 0:
            ax.text(0.5, 0.5, 'No misclassified examples found!',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return

        # Select diverse examples
        n_examples = min(20, len(misclassified_indices))
        sample_indices = np.random.choice(misclassified_indices, n_examples, replace=False)

        # Get images and create grid
        images = []
        labels_info = []
        for idx in sample_indices:
            img = self.test_dataset[idx][0].numpy().squeeze()
            images.append(img)
            true_lbl = res['labels'][idx]
            pred_lbl = res['predictions'][idx]
            confidence = res['probabilities'][idx].max()
            labels_info.append((true_lbl, pred_lbl, confidence))

        grid = self._create_image_grid(images, rows=4, cols=5)
        ax.imshow(grid, cmap='gray', interpolation='nearest')

        # Add labels with better formatting
        for i, (true_lbl, pred_lbl, conf) in enumerate(labels_info):
            row, col = i // 5, i % 5
            x, y = col * 28 + 14, row * 28 - 3

            # Color code based on confidence
            color = 'red' if conf > 0.8 else 'orange' if conf > 0.5 else 'darkred'

            ax.text(x, y, f'{true_lbl}→{pred_lbl}\n{conf:.2f}',
                   ha='center', va='bottom', fontsize=9, weight='bold',
                   color=color,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9))

    def _create_image_grid(self, images, rows=2, cols=5):
        """Helper to create a grid of images with improved spacing."""
        if len(images) == 0:
            return np.zeros((28*rows, 28*cols))

        h, w = 28, 28  # MNIST image dimensions
        grid = np.zeros((h * rows, w * cols))

        for i, img in enumerate(images[:rows*cols]):
            if img is None:
                continue
            r, c = i // cols, i % cols
            img_reshaped = img.reshape(h, w) if img.ndim == 1 else img
            grid[r*h:(r+1)*h, c*w:(c+1)*w] = img_reshaped

        return grid    
    
    def generate_summary_report(self):
        """Generate a comprehensive text summary of the analysis for both models."""
        print("\n" + "="*80)
        print("COMPREHENSIVE ANALYSIS SUMMARY REPORT - STANDARD vs YAT MODELS")
        print("="*80)

        # Model Performance Summary
        print("\n📊 MODEL PERFORMANCE COMPARISON:")
        print("-" * 40)
        model_names = [name for name in self.results if name not in ['training_history', 'yat_training_history', 'weight_analysis', 'weight_analysis_yat', 'similarity_correlations']]
        for name in sorted(model_names, key=lambda x: self.results[x]['accuracy'], reverse=True):
            acc = self.results[name]['accuracy']
            marker = " ⭐" if name in ['trained', 'trained_yat'] else ""
            print(f"{name.replace('_', ' ').title():20s}: {acc:.4f} ({acc*100:.2f}%){marker}")

        # Direct comparison
        standard_acc = self.results['trained']['accuracy']
        yat_acc = self.results['trained_yat']['accuracy']
        diff = yat_acc - standard_acc
        winner = "YAT" if diff > 0 else "Standard" if diff < 0 else "Tie"
        
        print(f"\n🏆 DIRECT COMPARISON (Standard vs YAT):")
        print("-" * 40)
        print(f"Standard Model:  {standard_acc:.4f} ({standard_acc*100:.2f}%)")
        print(f"YAT Model:       {yat_acc:.4f} ({yat_acc*100:.2f}%)")
        print(f"Difference:      {diff:+.4f} ({diff*100:+.2f}%)")
        print(f"Winner:          {winner} Model")

        # Similarity Analysis
        if 'similarity_correlations' in self.results:
            print(f"\n🔍 SIMILARITY MEASURE CORRELATIONS:")
            print("-" * 40)
            correlations = self.results['similarity_correlations']
            for pair, data in correlations.items():
                corr = data['correlation']
                significance = "***" if data['p_value'] < 0.001 else "**" if data['p_value'] < 0.01 else "*" if data['p_value'] < 0.05 else ""
                print(f"{pair:25s}: {corr:6.3f} {significance}")

        # Weight Analysis Comparison
        print(f"\n⚖️  WEIGHT STRUCTURE ANALYSIS:")
        print("-" * 40)
        
        # Standard model weights
        standard_weight_norms = self.results['weight_analysis']['weight_norms']
        standard_proto_norms = self.results['weight_analysis']['prototype_norms']
        
        # YAT model weights  
        yat_weight_norms = self.results['weight_analysis_yat']['weight_norms']
        yat_proto_norms = self.results['weight_analysis_yat']['prototype_norms']
        
        print(f"STANDARD MODEL:")
        print(f"  Weight Norms   - Mean: {standard_weight_norms.mean():.3f}, Std: {standard_weight_norms.std():.3f}")
        print(f"  Prototype Norms - Mean: {standard_proto_norms.mean():.3f}, Std: {standard_proto_norms.std():.3f}")
        
        print(f"YAT MODEL:")
        print(f"  Weight Norms   - Mean: {yat_weight_norms.mean():.3f}, Std: {yat_weight_norms.std():.3f}")
        print(f"  Prototype Norms - Mean: {yat_proto_norms.mean():.3f}, Std: {yat_proto_norms.std():.3f}")

        # Training Summary Comparison
        print(f"\n🎯 TRAINING SUMMARY COMPARISON:")
        print("-" * 40)
        
        # Standard model training
        standard_history = self.results['training_history']
        standard_final_train_acc = standard_history['train_acc'][-1]
        standard_final_val_acc = standard_history['val_acc'][-1]
        standard_epochs = len(standard_history['train_loss'])
        
        # YAT model training
        yat_history = self.results['yat_training_history']
        yat_final_train_acc = yat_history['train_acc'][-1]
        yat_final_val_acc = yat_history['val_acc'][-1]
        yat_epochs = len(yat_history['train_loss'])
        
        print(f"STANDARD MODEL:")
        print(f"  Epochs Trained: {standard_epochs}")
        print(f"  Final Train Accuracy: {standard_final_train_acc:.4f}")
        print(f"  Final Validation Accuracy: {standard_final_val_acc:.4f}")
        print(f"  Test Accuracy: {standard_acc:.4f}")
        
        print(f"YAT MODEL:")
        print(f"  Epochs Trained: {yat_epochs}")
        print(f"  Final Train Accuracy: {yat_final_train_acc:.4f}")
        print(f"  Final Validation Accuracy: {yat_final_val_acc:.4f}")
        print(f"  Test Accuracy: {yat_acc:.4f}")

        # Error Analysis
        print(f"\n🔍 ERROR ANALYSIS:")
        print("-" * 40)
        
        standard_cm = self.results['trained']['confusion_matrix']
        yat_cm = self.results['trained_yat']['confusion_matrix']
        
        standard_errors = standard_cm.sum() - np.trace(standard_cm)
        yat_errors = yat_cm.sum() - np.trace(yat_cm)
        
        print(f"Standard Model Errors: {standard_errors}")
        print(f"YAT Model Errors:      {yat_errors}")
        print(f"Error Difference:      {yat_errors - standard_errors:+d}")

        print("\n" + "="*80)      
    def run_enhanced_experiment(self, visualize=True, advanced_analysis=True, save_results=True):
        """Executes the full enhanced experimental pipeline with both models."""
        print("🚀 Starting Enhanced MNIST Single-Layer Analysis (Standard vs YAT)...")
        self.load_data()
        self.train()
        self.analyze()
        
        if advanced_analysis:
            print("\n🔬 Running advanced analysis...")
            self.evaluate_model_robustness()
            self.analyze_performance_by_difficulty()
        
        self.generate_summary_report()
        
        if advanced_analysis:
            self.generate_detailed_comparison_report()

        if visualize:
            print("\n📊 Generating visualizations...")
            self.visualize_training_performance()
            self.visualize_weights_and_prototypes()
            self.visualize_similarity_analysis()
            self.visualize_comprehensive_model_comparison()
            
            if advanced_analysis:
                self.visualize_advanced_model_analysis()
                self.visualize_robustness_comparison()
                self.visualize_difficulty_analysis()
                self.visualize_prediction_comparison_gallery()
            
            self.visualize_confusion_matrices_comparison()
            self.visualize_dimensionality_reduction()
            self.visualize_statistical_analysis()
            self.visualize_detailed_performance()
            self.visualize_error_analysis()

        if save_results:
            print("\n💾 Saving results...")
            self.save_experiment_results()
            self.create_comprehensive_report()

        print("\n✅ Enhanced experiment completed successfully!")
        return self.results

    def run_quick_experiment(self):
        """Run a quick experiment with minimal visualizations."""
        print("🚀 Starting Quick MNIST Analysis (Standard vs YAT)...")
        self.load_data()
        self.train()
        self.analyze()
        self.generate_summary_report()
        
        # Only essential visualizations
        self.visualize_training_performance()
        self.visualize_comprehensive_model_comparison()
        
        print("\n✅ Quick experiment completed!")
        return self.results

if __name__ == '__main__':
    print("MNIST Standard vs YAT Classifier Comparison")
    print("=" * 50)
    print("Choose experiment type:")
    print("1. Quick experiment (basic comparison)")
    print("2. Full experiment (all analyses and visualizations)")
    print("3. Custom experiment")
    
    try:
        choice = input("\nEnter your choice (1-3) [default: 2]: ").strip()
        if not choice:
            choice = "2"
    except:
        choice = "2"
    
    experiment = EnhancedSingleLayerExperiment()
    
    if choice == "1":
        results = experiment.run_quick_experiment()
    elif choice == "3":
        print("\nCustom experiment options:")
        visualize = input("Generate visualizations? (y/n) [default: y]: ").strip().lower() != 'n'
        advanced = input("Run advanced analysis? (y/n) [default: y]: ").strip().lower() != 'n'
        save = input("Save results? (y/n) [default: y]: ").strip().lower() != 'n'
        
        results = experiment.run_enhanced_experiment(
            visualize=visualize, 
            advanced_analysis=advanced, 
            save_results=save
        )
    else:  # Default: full experiment
        results = experiment.run_enhanced_experiment()
    
    print(f"\n🎉 Experiment completed! Final comparison:")
    print(f"Standard Model Accuracy: {results['trained']['accuracy']:.4f}")
    print(f"YAT Model Accuracy:      {results['trained_yat']['accuracy']:.4f}")
    diff = results['trained_yat']['accuracy'] - results['trained']['accuracy']
    winner = "YAT" if diff > 0 else "Standard" if diff < 0 else "Neither (tie)"
    print(f"Winner: {winner} (difference: {diff:+.4f})")

    def visualize_confusion_matrices_comparison(self):
        """Creates a dedicated visualization comparing all model confusion matrices."""
        print("Generating confusion matrices comparison...")

        # Get all available models
        model_names = [name for name in self.results if name not in ['training_history', 'weight_analysis', 'similarity_correlations']]
        model_names = sorted(model_names, key=lambda x: self.results[x]['accuracy'], reverse=True)

        # Calculate grid dimensions
        n_models = len(model_names)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols

        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), constrained_layout=True)
        fig.suptitle('Confusion Matrices Comparison - All Models', fontsize=20, weight='bold', y=0.98)

        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)

        # Plot confusion matrices
        for i, model_name in enumerate(model_names):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            self._plot_confusion_matrix_unified(ax, model_name)

        # Hide unused subplots
        for i in range(n_models, rows * cols):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.axis('off')

        plt.show()

    def visualize_confusion_matrix_detailed(self, model_name='trained'):
        """Creates a detailed view of a single confusion matrix with enhanced statistics."""
        if model_name not in self.results:
            print(f"Model '{model_name}' not found in results.")
            return

        print(f"Generating detailed confusion matrix for {model_name}...")

        cm = self.results[model_name]['confusion_matrix']
        acc = self.results[model_name]['accuracy']

        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
        fig.suptitle(f'Detailed Confusion Matrix Analysis - {model_name.replace("_", " ").title()}',
                    fontsize=18, weight='bold')

        # 1. Standard confusion matrix with counts and percentages
        self._plot_confusion_matrix_unified(ax1, model_name, ' - Counts & Percentages')

        # 2. Normalized confusion matrix (row percentages)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax2,
                   cbar_kws={'label': 'Percentage'}, square=True)
        ax2.set_title('Row-Normalized Confusion Matrix\n(Recall per Class)', fontsize=12, weight='bold')
        ax2.set_xlabel('Predicted Class')
        ax2.set_ylabel('True Class')

        # 3. Column-normalized confusion matrix (precision)
        cm_norm_col = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        sns.heatmap(cm_norm_col, annot=True, fmt='.2f', cmap='Greens', ax=ax3,
                   cbar_kws={'label': 'Percentage'}, square=True)
        ax3.set_title('Column-Normalized Confusion Matrix\n(Precision per Class)', fontsize=12, weight='bold')
        ax3.set_xlabel('Predicted Class')
        ax3.set_ylabel('True Class')

        # 4. Error analysis
        ax4.axis('off')

        # Calculate detailed statistics
        report = self.results[model_name]['classification_report']

        # Per-class statistics
        class_stats = []
        for i in range(10):
            class_name = str(i)
            precision = report[class_name]['precision']
            recall = report[class_name]['recall']
            f1 = report[class_name]['f1-score']
            support = report[class_name]['support']
            class_stats.append(f"Class {i}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, N={support}")

        # Most confused pairs
        confused_pairs = []
        for i in range(10):
            for j in range(10):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append((cm[i, j], i, j))
        confused_pairs.sort(reverse=True)
        top_confusions = confused_pairs[:5]

        analysis_text = f"""DETAILED PERFORMANCE ANALYSIS

Overall Accuracy: {acc:.4f} ({acc*100:.2f}%)

PER-CLASS METRICS:
{chr(10).join(class_stats)}

MACRO AVERAGES:
Precision: {report['macro avg']['precision']:.3f}
Recall: {report['macro avg']['recall']:.3f}
F1-Score: {report['macro avg']['f1-score']:.3f}

MOST COMMON CONFUSIONS:
{chr(10).join([f"{true} → {pred}: {count} times" for count, true, pred in top_confusions])}

TOTAL SAMPLES: {cm.sum()}
TOTAL ERRORS: {cm.sum() - np.trace(cm)}
ERROR RATE: {(1-acc)*100:.2f}%"""

        ax4.text(0.05, 0.95, analysis_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='left', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

        plt.show()
    def analyze_performance_by_difficulty(self):
        """Analyze model performance on easy vs hard samples."""
        print("Analyzing performance by sample difficulty...")
        
        # Define difficulty based on class prototypes similarity
        prototypes = {i: self.class_stats[i]['mean'] for i in range(10)}
        
        difficulties = []
        true_labels = []
        
        # Calculate difficulty for each test sample
        for i, (data, label) in enumerate(self.test_dataset):
            if i >= 1000:  # Limit for performance
                break
                
            sample = data.view(-1).numpy()
            prototype = prototypes[label]
            
            # Difficulty = inverse of similarity to correct prototype
            similarity = self._cosine_similarity(sample, prototype)
            difficulty = 1 - similarity
            
            difficulties.append(difficulty)
            true_labels.append(label)
        
        difficulties = np.array(difficulties)
        true_labels = np.array(true_labels)
        
        # Define easy/hard thresholds
        easy_threshold = np.percentile(difficulties, 33)
        hard_threshold = np.percentile(difficulties, 67)
        
        easy_mask = difficulties <= easy_threshold
        medium_mask = (difficulties > easy_threshold) & (difficulties <= hard_threshold)
        hard_mask = difficulties > hard_threshold
        
        # Evaluate models on each subset
        subsets = {
            'Easy': easy_mask,
            'Medium': medium_mask,
            'Hard': hard_mask
        }
        
        print(f"\nDifficulty analysis on {len(difficulties)} samples:")
        print(f"Easy samples (≤{easy_threshold:.3f}): {np.sum(easy_mask)}")
        print(f"Medium samples: {np.sum(medium_mask)}")
        print(f"Hard samples (≥{hard_threshold:.3f}): {np.sum(hard_mask)}")
        
        results_by_difficulty = {}
        
        for subset_name, mask in subsets.items():
            subset_indices = np.where(mask)[0]
            
            if len(subset_indices) == 0:
                continue
            
            # Get predictions for this subset
            std_preds_subset = self.results['trained']['predictions'][:len(difficulties)][mask]
            yat_preds_subset = self.results['trained_yat']['predictions'][:len(difficulties)][mask]
            true_labels_subset = true_labels[mask]
            
            std_acc = np.mean(std_preds_subset == true_labels_subset)
            yat_acc = np.mean(yat_preds_subset == true_labels_subset)
            
            results_by_difficulty[subset_name] = {
                'standard_accuracy': std_acc,
                'yat_accuracy': yat_acc,
                'sample_count': len(subset_indices),
                'difficulty_range': (np.min(difficulties[mask]), np.max(difficulties[mask]))
            }
            
            print(f"\n{subset_name} samples:")
            print(f"  Standard accuracy: {std_acc:.4f}")
            print(f"  YAT accuracy: {yat_acc:.4f}")
            print(f"  Difference: {yat_acc - std_acc:+.4f}")
        
        self.results['difficulty_analysis'] = results_by_difficulty
        return results_by_difficulty

    def visualize_difficulty_analysis(self):
        """Visualize performance by sample difficulty."""
        if 'difficulty_analysis' not in self.results:
            self.analyze_performance_by_difficulty()
        
        print("Generating difficulty analysis visualization...")
        plt.style.use('seaborn-v0_8-whitegrid')
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
        fig.suptitle('Performance Analysis by Sample Difficulty', fontsize=18, weight='bold')
        
        difficulty_data = self.results['difficulty_analysis']
        subsets = list(difficulty_data.keys())
        
        std_accs = [difficulty_data[subset]['standard_accuracy'] for subset in subsets]
        yat_accs = [difficulty_data[subset]['yat_accuracy'] for subset in subsets]
        sample_counts = [difficulty_data[subset]['sample_count'] for subset in subsets]
        
        # Plot 1: Accuracy by difficulty
        x = np.arange(len(subsets))
        width = 0.35
        
        bars1 = axes[0].bar(x - width/2, std_accs, width, label='Standard', alpha=0.8, color='blue')
        bars2 = axes[0].bar(x + width/2, yat_accs, width, label='YAT', alpha=0.8, color='red')
        
        axes[0].set_title('Accuracy by Sample Difficulty', fontsize=14, weight='bold')
        axes[0].set_xlabel('Difficulty Level')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(subsets)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].set_ylim(0, 1)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Plot 2: Performance difference
        differences = [yat_acc - std_acc for std_acc, yat_acc in zip(std_accs, yat_accs)]
        colors = ['green' if diff > 0 else 'red' if diff < 0 else 'gray' for diff in differences]
        
        bars = axes[1].bar(subsets, differences, color=colors, alpha=0.7, edgecolor='black')
        axes[1].set_title('YAT Advantage by Difficulty\n(YAT - Standard)', fontsize=14, weight='bold')
        axes[1].set_xlabel('Difficulty Level')
        axes[1].set_ylabel('Accuracy Difference')
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, diff in zip(bars, differences):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., 
                        height + (0.005 if height >= 0 else -0.005),
                        f'{diff:+.3f}', ha='center', 
                        va='bottom' if height >= 0 else 'top', 
                        fontweight='bold', fontsize=10)
        
        plt.show()
        
        # Print summary
        print(f"\nDifficulty Analysis Summary:")
        for subset in subsets:
            data = difficulty_data[subset]
            advantage = "YAT" if data['yat_accuracy'] > data['standard_accuracy'] else "Standard"
            margin = abs(data['yat_accuracy'] - data['standard_accuracy'])
            print(f"  {subset}: {advantage} wins by {margin:.3f} ({data['sample_count']} samples)")

    def create_comprehensive_report(self):
        """Create a comprehensive HTML report of all analyses."""
        print("Generating comprehensive HTML report...")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MNIST Standard vs YAT Classifier Comparison</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 10px; }}
                .section {{ margin: 30px 0; padding: 20px; border-left: 4px solid #007acc; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background-color: #f9f9f9; border-radius: 5px; }}
                .winner {{ color: #2e7d32; font-weight: bold; }}
                .loser {{ color: #c62828; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
                th {{ background-color: #f2f2f2; }}
                .improvement {{ background-color: #e8f5e8; }}
                .degradation {{ background-color: #ffe8e8; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>MNIST Classifier Comparison Report</h1>
                <h2>Standard Linear Classifier vs YAT Neural Module</h2>
                <p><strong>Generated:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        # Add main results
        std_acc = self.results['trained']['accuracy']
        yat_acc = self.results['trained_yat']['accuracy']
        
        html_content += f"""
            <div class="section">
                <h2>📊 Main Results</h2>
                <div class="metric">
                    <h3>Standard Model</h3>
                    <p>Accuracy: <span class="{'winner' if std_acc > yat_acc else 'loser'}">{std_acc:.4f}</span></p>
                </div>
                <div class="metric">
                    <h3>YAT Model</h3>
                    <p>Accuracy: <span class="{'winner' if yat_acc > std_acc else 'loser'}">{yat_acc:.4f}</span></p>
                </div>
                <div class="metric">
                    <h3>Difference</h3>
                    <p><span class="{'winner' if yat_acc > std_acc else 'loser'}">{yat_acc - std_acc:+.4f}</span></p>
                </div>
            </div>
        """
        
        # Add per-class comparison if available
        if 'trained' in self.results and 'trained_yat' in self.results:
            std_report = self.results['trained']['classification_report']
            yat_report = self.results['trained_yat']['classification_report']
            
            html_content += """
                <div class="section">
                    <h2>🎯 Per-Class Performance</h2>
                    <table>
                        <tr>
                            <th>Class</th>
                            <th>Standard F1</th>
                            <th>YAT F1</th>
                            <th>Difference</th>
                            <th>Winner</th>
                        </tr>
            """
            
            for i in range(10):
                std_f1 = std_report[str(i)]['f1-score']
                yat_f1 = yat_report[str(i)]['f1-score']
                diff = yat_f1 - std_f1
                winner = "YAT" if diff > 0.001 else "Standard" if diff < -0.001 else "Tie"
                row_class = "improvement" if diff > 0.001 else "degradation" if diff < -0.001 else ""
                
                html_content += f"""
                        <tr class="{row_class}">
                            <td>{i}</td>
                            <td>{std_f1:.4f}</td>
                            <td>{yat_f1:.4f}</td>
                            <td>{diff:+.4f}</td>
                            <td>{winner}</td>
                        </tr>
                """
            
            html_content += """
                    </table>
                </div>
            """
        
        # Add robustness results if available
        if 'robustness_test' in self.results:
            robustness = self.results['robustness_test']
            html_content += """
                <div class="section">
                    <h2>🛡️ Robustness Analysis</h2>
                    <table>
                        <tr>
                            <th>Noise Level</th>
                            <th>Standard Accuracy</th>
                            <th>YAT Accuracy</th>
                            <th>Difference</th>
                        </tr>
            """
            
            for noise, std_acc, yat_acc in zip(robustness['noise_levels'], 
                                              robustness['standard_accuracies'], 
                                              robustness['yat_accuracies']):
                diff = yat_acc - std_acc
                row_class = "improvement" if diff > 0.001 else "degradation" if diff < -0.001 else ""
                
                html_content += f"""
                        <tr class="{row_class}">
                            <td>{noise:.1f}</td>
                            <td>{std_acc:.4f}</td>
                            <td>{yat_acc:.4f}</td>
                            <td>{diff:+.4f}</td>
                        </tr>
                """
            
            html_content += """
                    </table>
                </div>
            """
        
        html_content += """
            </body>
            </html>
        """
        
        # Save HTML report
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mnist_comparison_report_{timestamp}.html"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"HTML report saved to: {filename}")
            return filename
        except Exception as e:
            print(f"Error saving HTML report: {e}")
            return None