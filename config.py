"""
Default configuration for the experiment.
"""

def get_default_config():
    """Returns the default configuration dictionary."""
    return {
        'seed': 42,
        'batch_size': 128,
        'validation_split': 0.1,
        'num_epochs': 25,
        'lr': 0.01,
        'weight_decay': 1e-4,
        'scheduler_step': 10,
        'patience': 5,
        'dropout': 0.1
    }
