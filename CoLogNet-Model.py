import sys
import pandas as pd #loading data in table form
import numpy as np # linear algebra
import torch # import main library
import torch.nn as nn # import modules
from torch.autograd import Function # import Function to create custom activations
from torch.nn.parameter import Parameter # import Parameter to create custom activations with learnable parameters
import torch.nn.functional as F # import torch functions

from enum import Enum

#Straight-Through Estimator
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

#Continued Fraction/Logarithm Model

class Variant(Enum):
    COFRNET = "cofr"
    COLOGNET = "colog"
    COLOGNET_B = "colog_b" 

class ContNet_Model(nn.Module):
    '''
    Variant 1: CoFrNet
    - Continued fractions, continuant form
    - Floating point

    Variant 2: CoLogNet
    - Continued logarithms, continuant form
    - Floating point

    Variant 3: CoLogNetB
    - Continued logarithms, continuant form
    - Outputs binarized before recurrence formula

    All variants follow Ensemble-Ladder Structure: 
        each Output dimension has one Ensemble
        each Ensemble is made up of *num_ladders* ladders
        each Ladder has *depth* rungs or values

        1) Linear Layer performed on ladder values
        2) Ladder values passed into Recurrence Formula
        3) Outputs for each ladder averaged to get one output per Ensemble
    '''
    @classmethod
    def __init__(cls, model_type, input_size, output_size, depth, num_ladders, dropout):
        depth_list = np.full(depth, num_ladders)
        return cls(model_type, input_size, output_size, depth_list, dropout)

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

        for i in range(0, output_size): #create an ensemble for each output dim
            ensemble = Ensemble(model_type, input_size, output_size, depth_list, dropout)
            self.layers.append(ensemble)

    def forward(self, inputs):
        '''
        Docstring for forward

        :param inputs: 2D input vector
        '''
        final_vector = [ensemble(inputs) for ensemble in self.layers]
        final_vector = torch.stack(final_vector, dim=1)  # combine efficiently  
        
        return final_vector  # shape: (batch_size, output_size)

class Ensemble(nn.Module):
    def __init__(self, model_type, input_size, output_size, depth_list, dropout):

        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.depth_list = depth_list
        self.ladders = nn.ModuleList()
        for depth in depth_list: #for each ladder in ensemble
            ladder = Ladder(model_type, input_size, depth, dropout)
            self.ladders.append(ladder) #add new ladder to ladders

    def forward(self, inputs):
        final_vector = [ladder(inputs) for ladder in self.ladders]
        output = torch.stack(final_vector)
        output = output.mean(dim=0)
       
        return output
    
class Ladder(nn.Module):
    def __init__(self, model_type, input_size, depth, dropout):
        super().__init__()
        self.model_type = model_type
        self.input_size = input_size
        self.depth = depth
        self.dropout = nn.Dropout(p=dropout) # optional dropout

        self.weight = nn.Parameter(torch.Tensor(depth, input_size)) #depth rows & input_size columns
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight) 

    def forward(self, inputs):
        linear_output = F.linear(inputs, self.weight) #linear layer, output: [batch_size, depth]
        transposed_output = torch.transpose(linear_output, 0, 1) #flips rows and columns, output: [depth, batch_size]

        if self.model_type is Variant.COLOGNET_B: #if model is binarized, use STE
            output = binarize_with_ste(transposed_output)
        else:
            output = torch.sigmoid(transposed_output)

        recc_form_output = self.recurrence_formula(output, self.model_type) #apply recurrence formula
        output = self.dropout(recc_form_output)
        return output
    
    @staticmethod
    def recurrence_formula(X: torch.Tensor, model_type): 
        #X: tensor of exponents for recurrence terms, model_type: Variant.COFRNET, Variant.COLOGNET, Variant.COLOGNETB
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
    
        n = len(X)

        if model_type is Variant.COFRNET:
            C = X
        else:
            C = torch.pow(2, X) #raise 2^x for each value

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
        
        return A_n / B_n