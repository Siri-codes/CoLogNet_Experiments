import unittest
import os
import shutil
import sys

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
import torch # import main library
import torch.nn as nn # import modules
from torch.autograd import Function # import Function to create custom activations
from torch.nn.parameter import Parameter # import Parameter to create custom activations with learnable parameters
import torch.nn.functional as F # import torch functions
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch.optim as optim
from sklearn.datasets import load_breast_cancer, load_wine, load_linnerud, load_diabetes, load_iris, load_digits
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import random
import pandas as pd
import matplotlib.pyplot as plt

def train_test_loop(dataset, data_dir, model_type, depth, num_ladders, learning_rate, dropout, batch_size):

  train_loader, val_loader, test_loader = process_data(dataset, data_dir, batch_size)

  if dataset is Dataset_Enum.MNIST:
    input_size = 784
    output_size = 10
  elif dataset is Dataset_Enum.CIFAR10:
    input_size = 3072
    output_size = 10

  model = ContNet_Model(model_type, input_size, output_size, depth, num_ladders, dropout)

  train(model, train_loader, val_loader, num_epochs=20, lr=learning_rate, weight_decay=1e-4)

  total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) #optional: print total parameters of model
  #print("Total trainable parameters:", total_params)#

  accuracy = get_accuracy(model, test_loader)

  #print("Accuracy:", accuracy)#
  return accuracy, total_params

def get_accuracy(self, dataloader):
    self.model.eval()  # Set model to evaluation mode
    num_correct = 0
    num_total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(self.model.device)  # Move to same device as model
            labels = labels.to(self.model.device)

            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get class with highest probability

            num_correct += (predicted == labels).sum().item()
            num_total += labels.size(0)

    accuracy = num_correct / num_total
    return accuracy