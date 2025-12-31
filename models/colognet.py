
#COFRNET/COLOGNET MODEL DEFINITIONS

import torch # import main library
import torch.nn as nn # import modules
from torch.autograd import Function # import Function to create custom activations
from torch.nn.parameter import Parameter # import Parameter to create custom activations with learnable parameters
import torch.nn.functional as F # import torch functions

from enum import Enum

#Straight-Through Estimator (for binarized CoLogNetB)
class Binarize_STE(torch.autograd.Function): 
    '''
    A typical straight through estimator for BNNs (e.g. XNOR-Net)
    Binarizes values to -1 or 1 and clips gradients in backward pass.
    '''
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input) #binarized to -1 or 1
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clamp(-1, 1) # clip gradient between -1 and 1, since sign gradient is typically flat
        return grad_input
    
def binarize_with_ste(x):
    return Binarize_STE.apply(x)

class Euclidean_Distance(nn.Module):
    def __init__(self, in_features, out_features):
        def __init__(self, in_features, out_features):
        super().__init__()

        # like linear layer, has learnable weights (represent centroids?)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        # common initialization for custom layers
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        # x shape: (batch, in_features)
        # weight shape: (out_features, in_features)
        # unsqueeze to match cdist dims
        # output shape: (batch, out_features)
        return torch.cdist(x.unsqueeze(1), self.weight.unsqueeze(0), p=2).squeeze(1)

#Continued Fraction/Logarithm Model Variants
class Variant(Enum):
    COFRNET = "cofr"
    COLOGNET = "colog"
    COLOGNET_B = "colog_b" 
    COLOGNET_E = "colog_e"
    MLP = "mlp"
    SWIGLU = "swiglu"

#Continued Fraction/Logarithm Neural Network Model
class ContNet_Model(nn.Module):
    '''
    Variant 1: CoFrNet
    - Continued fractions, continuant form
    - Floating point

    Variant 2: CoLogNet
    - Continued logarithms, continuant form
    - Floating point

    Variant 3: CoLogNet-B
    - Continued logarithms, continuant form
    - Outputs binarized before recurrence formula

    Variant 4: CoLogNet-E
    - Continued logarithms, continuant form
    - Floating point
    - Euclidean distance instead of dot product with weights to get coefficients

    All variants follow Ladder Structure:
        Given a list of depths [d_1, d_2, d_3, ..., d_n], creates n ladders

        For each ladder,
        1) matrix multiply inputs by weights w_i to get coefficients
        2) apply recurrence formula using coefficients --> single value

        To combine ladder results, perform matrix multiply with final_layer to get vector of output_size
    '''
    
    def __init__(self, model_type, input_size, output_size, depth_list, dropout):
        '''
        Docstring for __init__
        
        :param model_type: Variant.COFRNET, Variant.COLOGNET, or Variant.COLOGNETB
        :param input_size: size of input data
        :param output_size: size of output vector
        :param depth_list: list of depths of ladders
        :param dropout: dropout rate to prevent overfitting, typical values between 0.2 and 0.5
        
        '''
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.depth_list = depth_list
        self.layers = nn.ModuleList()

        self.final_layer = nn.Linear(in_features=len(depth_list), out_features=output_size, bias=True)

        for d_i in depth_list:
          l_i = Ladder(model_type, input_size, d_i, dropout)
          self.layers.append(l_i)

    def forward(self, inputs):
        '''
        Docstring for forward

        :param inputs: 2D input vector [input_size, batch_size]
        '''
        # Get outputs from each ladder
        output = torch.stack([ladder(inputs) for ladder in self.layers], dim=1)

        # Apply final linear layer
        output = self.final_layer(output)

        # Depending on whether this is a regression task, may need different formatting:
        return output.view(-1) if self.output_size == 1 else output
    
class Ladder(nn.Module):
    def __init__(self, model_type, input_size, depth, dropout):
        super().__init__()
        self.model_type = model_type
        self.input_size = input_size
        self.depth = depth

        self.weight = nn.Linear(input_size, depth) #linear layer to get coefficients
        self.norm = nn.LayerNorm(depth) #normalization layer
        self.dropout = nn.Dropout(p=dropout) # optional dropout layer
        
    def forward(self, inputs):
        # Coefficients: [batch, depth]
      coeffs = self.norm(self.weight(inputs))

      # Binarize if CoLogB
      if self.model_type is Variant.COLOGNET_B:
        coeffs = binarize_with_ste(coeffs)

      # Transpose for recurrence: [depth, batch]
      coeffs_t = coeffs.t()

      # Recurrence --> [batch]
      output = self.recurrence_formula(coeffs_t, self.model_type)

      # Apply dropout
      output = self.dropout(output.view(-1, 1)) # ensuring output is one column of length batch_size

      # Return as [batch] for stacking
      return output.view(-1)
    
    @staticmethod
    def recurrence_formula(X: torch.Tensor, model_type): 
        # X: tensor of exponents for recurrence terms, 
        # model_type: Variant.COFRNET, Variant.COLOGNET, Variant.COLOGNETB
        '''
        Computes convergent x_n = A_n/B_n, where
        
        A_0 = C_0
        A_1 = (C_1 * C_0) + C_0
        A_n = 
            - (C_n * A_{n-1}) + A_{n-2} (Continued Fractions) or
            - (C_n * A_{n-1}) + (C_{n-1} * A_{n-2}) (Continued Logarithms)

        B_0 = 1
        B_1 = C_1
        B_n = 
            - (C_n * B_{n-1}) + B{n-2} (Continued Fractions) or
            - (C_n * B_{n-1}) + (C_{n-1} * B{n-2}) (Continued Logarithms)

        and C_i is either:
            - X[i] if Continued Fractions
            - 2^(X[i]) if Continued Logarithms

        '''
    
        n = X.shape[0]

        if model_type is Variant.COFRNET:
            C = X
            epsilon = 0.1
        else:
            #(X, -10, 10) #clamp exponents between -10 and 10 for stability -- can change range if needed
            C = torch.pow(2, X) #raise 2^x for each value
            epsilon = 0.0

        A_neg1 = 1.0
        B_neg1 = 0.0
        A_0 = C[0]
        B_0 = 1.0
        
        #base cases:
        if n == 1:
            return A_0 / B_0
        
        A_prev2 = A_neg1 # A{n-2} starts at A{-1}
        A_prev = A_0 # A{n-1} starts at A{0}
        B_prev2 = B_neg1 # B{n-2} starts at B{-1}
        B_prev = B_0  # B{n-1} starts at B{0}

        #recursive step (A_n and B_n formulas dependent on model_type)
        
        for i in range(1, n):
            if model_type is Variant.COFRNET:
                dep_term = 1.0
            else:
                dep_term = C[i-1]

            A_n = C[i] * A_prev + dep_term * A_prev2 #next numerator term
            B_n = C[i] * B_prev + dep_term * B_prev2 #next denominator term
            A_prev2, A_prev = A_prev, A_n #prepping new A_{n-2} and A{n-1} terms
            B_prev2, B_prev = B_prev, B_n #ditto

        return A_n / (B_n + epsilon) #add small epsilon if cofrnet