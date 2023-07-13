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

class SimpleCNN(AbstractNN):
    """
    A simple Convolutional Neural Network (CNN) model.
    """
    def __init__(self, short_name):
        super(SimpleCNN, self).__init__(short_name)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = nn.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output


class SimpleRNN(AbstractNN):
    """
    A simple Recurrent Neural Network (RNN) model.
    """
    def __init__(self, short_name, input_size=28, hidden_size=128, num_layers=2, num_classes=10):
        super(SimpleRNN, self).__init__(short_name)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*28, num_classes)
    
    def forward(self, x):
        # Set an initial hidden and cell state
        h0 = nn.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
        
        # Forward propagate the RNN
        out, _ = self.rnn(x, h0)
        
        # Pass the output of the last time step to the classifier
        out = self.fc(out[:, -1, :])
        return out
