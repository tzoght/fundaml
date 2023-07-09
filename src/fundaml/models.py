import torch.nn as nn
from abc import ABC, abstractmethod
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch import Tensor
from typing import Tuple
from torch.utils.tensorboard import SummaryWriter


class AbstractNN(nn.Module):
    """
    An abstract base class for all neural network models.
    """    
    def __init__(self,short_name):
        super(AbstractNN,self).__init__()
        self.short_name = short_name
    

class SampleNNClassifier(AbstractNN):
    """
    A classifier model built upon a multilayer perceptron.
    """    
    def __init__(self, short_name):
        super(SampleNNClassifier, self).__init__(short_name)
        
        # Layer to flatten the input data
        self.flatten = nn.Flatten()
        
        # Linear layers stacked together with ReLU activation functions
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),  # First hidden layer
            nn.ReLU(),
            nn.Linear(512, 512),    # Second hidden layer
            nn.ReLU(),
            nn.Linear(512, 10),     # Output layer
        )

    def forward(self, x):
        # Flatten the input data
        x = self.flatten(x)
        
        # Pass the data through the linear layers and activations
        logits = self.linear_relu_stack(x)
        
        return logits