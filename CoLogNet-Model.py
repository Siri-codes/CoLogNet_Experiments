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

    def __init__(self, model_type, input_size, output_size, depth, num_ladders, dropout):

        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.num_ladders = num_ladders
        self.depth = depth #depth of each ladder
        self.layers = nn.ModuleList()

        for i in range(0, output_size): #create an ensemble for each output dim
            ensemble = Ensemble(model_type, input_size, output_size, depth, num_ladders, dropout)
            self.layers.append(ensemble)

    def forward(self, inputs):
        final_vector = [ensemble(inputs) for ensemble in self.layers]
        final_vector = torch.stack(final_vector, dim=1)  # combine efficiently  
        
        return final_vector  # shape: (batch_size, output_size)

class Ensemble(nn.Module):
    def __init__(self, model_type, input_size, output_size, depth, num_ladders, dropout):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.depth = depth
        self.num_ladders = num_ladders
        self.ladders = nn.ModuleList()
        for i in range(0, num_ladders): #for each ladder in ensemble
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

        if self.model_type is Variant.COLOGNET_B:
            output = binarize_with_ste(transposed_output)
        else:
            output = torch.sigmoid(transposed_output)

        recc_form_output = self.recurrence_formula(output, self.model_type)
        output = self.dropout(recc_form_output)
        return output
    
    @staticmethod
    def recurrence_formula(X: torch.Tensor, model_type):
        '''
        Computes convergent x_n = A_n/B_n, where
        
        A_0 = C_0
        A_1 = (C_1 * C_0) + C_0
        A_n = (C_n * A_{n-1}) + A_{n-2}

        B_0 = 1
        B_1 = C_1
        B_n = 
            - (C_n * B_{n-1}) + 1 (Continued Fractions) or
            - (C_n * B_{n-1}) + C_{n-1} (Continued Logarithms)

        and C_i is either:
            - X[i] if Continued Fractions
            - 2^(X[i]) if Continued Logarithms

        '''
    
        n = len(X)

        if model_type is Variant.COFRNET:
            C = X
        else:
            C = torch.pow(2, X) #raise 2^x for each value

    
        A_0 = C[0]
        B_0 = 1.0
        
        #base cases:
        if n == 1:
            return A_0 / B_0
        
        A_1 = C[1] * A_0 + A_0
        B_1 = C[1]
        if n == 2:
            return A_1 / B_1
        
        A_prev = A_0
        A_curr = A_1
        B_curr = B_1

        #recursive step (B_n formula dependent on model_type)
        
        for i in range(2, n):
            if model_type is Variant.COFRNET:
                B_term = 1
            else:
                B_term = C[i-1]

            A_next = C[i] * A_curr + A_prev #next numerator term
            B_next = C[i] * B_curr + B_term #next denominator term
            A_prev, A_curr = A_curr, A_next #prepping prev and curr to be used to calculate next term in future round
            B_curr = B_next
        
        return A_curr / B_curr