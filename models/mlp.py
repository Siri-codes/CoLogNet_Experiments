#MLP Models as Controls

import math
import torch.nn as nn 
import torch.nn.functional as F

class MLP(nn.Module):
  '''
  MLP neural network as control
  '''

  def __init__(self, input_size, output_size, depths, dropout=0.1):
      """
      Args:
          input_size (int): Input features.
          output_size (int): Output features.
          depths (list of int): List containing the depth for each ladder (e.g., [2, 3, 5, 8]).
          dropout (float): Dropout rate.
      """
      super().__init__()
      self.output_size = output_size

      # Calculate target parameters based on ContNet architecture
      target_params = calculate_parameters(input_size, output_size, depths)

      # Solve for MLP hidden layer size (h):
      # MLP params = h * (input_size + 1 + output_size) + output_size
      # --> h = (target_params - output_size) / (input_size + 1 + output_size)
      denom = input_size + 1 + output_size
      hidden_size = math.ceil((target_params - output_size) / denom) #for fair play, round up to give MLP the advantage

      if hidden_size <= 0: #hidden_size must be at least 1
          hidden_size = 1

      print(f"Total ContNet Parameters: {target_params}") #print target params for comparison
      print(f"Calculated MLP Hidden Size: {hidden_size}")

      self.mlp = nn.Sequential( #build MLP architecture
          nn.Linear(input_size, hidden_size),
          nn.BatchNorm1d(hidden_size),
          nn.ReLU(),
          nn.Dropout(p=dropout),
          nn.Linear(hidden_size, output_size)
      )

      # Initialization (Kaiming) for ReLU
      for m in self.modules():
          if isinstance(m, nn.Linear): 
              nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') 


      # Verification of parameters
      actual_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
      print(f"Actual MLP Parameters: {actual_params}")

  def forward(self, x):
        if x.dim() > 2: # flatten image inputs (MNIST) if needed
            x = x.view(x.size(0), -1)

        x = self.mlp(x)

        # squeeze for 1D regression targets (Boston Housing) if needed
        if self.output_size == 1:
            return x.squeeze(-1)
        return x

class SwiGLUMLP(nn.Module):
    def __init__(self, input_size, output_size, depths, dropout=0.1):
        super().__init__()
        self.output_size = output_size
        target_params = calculate_parameters(input_size, output_size, depths)

        # PARAMETER CALCULATION FOR SWIGLU:
        # Layer 1 (Up-projection): input_size * (2 * hidden) + (2 * hidden)
        # Layer 2 (Down-projection): hidden * output_size + (bias) output_size
        # Total = hidden * (2 * input_size + 2 + output_size) + output_size

        denom = (2 * input_size) + 2 + output_size
        hidden_size = math.ceil((target_params - output_size) / denom)

        if hidden_size <= 0: #hidden must be at least 1
            hidden_size = 1

        print(f"Target Parameters: {target_params}")
        print(f"Calculated SwiGLU Hidden Size (H): {hidden_size}")

        self.mlp = nn.Sequential(
            # We project to 2*hidden to facilitate the GLU split
            nn.Linear(input_size, 2 * hidden_size),
            SwiGLU(), #reduces dimensions from 2*hidden to hidden
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, output_size)
        )

        actual_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Actual SwiGLU MLP Parameters: {actual_params}")

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        x = self.mlp(x)

        if self.output_size == 1:
            return x.squeeze(-1)
        return x

class SwiGLU(nn.Module):
  """
  SwiGLU Activation Layer: (xW + b) * sigmoid(xV + c) * (xV + c)
  (Split input tensor in half along last dimension, apply SiLU (Swish) to one half, and multiply with the other half)
  """

  def forward(self, x):
      # Split the input tensor into two halves along the last dimension
      x, gate = x.chunk(2, dim=-1)
      return x * F.silu(gate) #apply swish func to gate side & multiply with non-gate side


@staticmethod
def calculate_parameters(input_size, output_size, depths):
  '''
  Calculates the total number of parameters for a ContNet model given input size, output size, and ladder depths.

  :param input_size: input size of the model
  :param output_size: output size of the model
  :param depths: list of ladder depths
  '''

  # 1. Calculate parameters for all ladders
  # Each ladder 'l_i' has a weight matrix of (input_size * depth_i) and bias of (depth_i)
  ladder_params_total = sum(d * input_size + d for d in depths)

  # 2. Calculate parameters for the final combiner
  # weight matrix: (output_size * sum(depths)), bias: (output_size)
  combiner_params = output_size * len(depths) + output_size

  # 3. Total Model Parameters
  target_params = ladder_params_total + combiner_params

  return target_params