import torch
import torch.nn as nn
import numpy as np
from nmn.torch.nmn import YatNMN

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
