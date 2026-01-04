
#COFRNET/COLOGNET MODEL DEFINITIONS

import torch # import main library
import torch.nn as nn # import modules
from torch.autograd import Function # import Function to create custom activations
from torch.nn.parameter import Parameter # import Parameter to create custom activations with learnable parameters
import torch.nn.functional as F # import torch functions

from enum import Enum
import math

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

class Euclidean_Distance_Layer(nn.Module):
    '''
    Custom layer to replace Linear Layer
    Calculates Euclidean / L2 distance between inputs and weights: d(x, w) = sqrt((x - w)^2)
    '''
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

#Continued Fraction/Logarithm Neural Network Model
class ContNet_Model(nn.Module):
    '''

    All variants follow Ladder Structure:
        Given a list of depths [d_1, d_2, d_3, ..., d_n], creates n ladders

        For each ladder (if parallel ladders),
        1) matrix multiply inputs by weights w_i to get coefficients (or use euclidean distance)
        2) apply recurrence formula using coefficients --> single value

        To combine ladder results, perform matrix multiply with final_layer to get vector of output_size

    '''
    
    def __init__(self, variant_settings: dict, input_size: int, output_size: int, depth_list: list, dropout: float):
        '''
        Docstring for __init__
        
        :param variant_settings: dictionary of settings for model variant
        :param input_size: size of input data
        :param output_size: size of output vector
        :param depth_list: list of depths of ladders
        :param dropout: dropout rate to prevent overfitting, typical values between 0.2 and 0.5
        
        '''
        super().__init__()
        
        #unpack variant settings
        self.is_cofr = variant_settings['is_cofr'] #using continued fractions (True) or continued logarithms (False)?
        self.is_bin = variant_settings['is_bin'] #binarizing coefficients before recurrence?
        self.is_orig = variant_settings['is_orig'] #using original formula instead of continuant form?
        self.is_seq = variant_settings['is_seq'] #using sequential ladders instead of parallel ones?
        self.is_euc_dist = variant_settings['is_euc_dist'] #using euclidean distance instead of dot product with weights to get coefficients?
        self.is_reciprocal = variant_settings['is_reciprocal'] #applying reciprocal after each ladder?
        self.is_dist_norm = variant_settings['is_dist_norm'] #normalizing coefficients to be unit vector?
        self.is_log = variant_settings['is_log'] #apply log before weights?
        self.is_mlp = variant_setting['is_mlp'] #using MLP instead of linear layer

        self.input_size = input_size
        self.output_size = output_size
        self.depth_list = depth_list
        self.layers = nn.ModuleList()
       
        for d_i in depth_list:
            l_i = Ladder(variant_settings, d_i, dropout)
            self.layers.append(l_i)

        self.final_layer = nn.Linear(in_features=len(depth_list), out_features=output_size, bias=True)
        
        #weights to get coefficients for recurrence formula
        if self.is_seq:
            self.coeff_weight = nn.Linear(input_size, depth_list[0]) #first ladder gets coefficients from inputs
        elif self.is_mlp:
            hidden_size = input_size 
            self.coeff_weight = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, sum(depth_list))
            )
        else:
            self.coeff_weight = Euclidean_Distance_Layer(input_size, sum(depth_list)) if self.is_euc_dist else nn.Linear(input_size, sum(depth_list))

        if self.is_seq:
            # Calculate the total number of filler terms needed across all ladders
            # skip first ladder, each ladder needs filler_size d_i - 1
            self.filler_sizes = [d_i - 1 for d_i in depth_list[1:]]
            self.total_filler_dim = sum(self.filler_sizes)
            
            # To save on parameters:
            rank = 16  # Hyperparameter: much smaller than input_size (784)
            self.res_con_down = nn.Linear(input_size, rank, bias=False)
            self.res_con_up = nn.Linear(rank, self.total_filler_dim, bias=True)

            # One big layer instead of many small ones
            self.res_con_net = nn.Linear(input_size, self.total_filler_dim)

    def forward(self, inputs):
        '''
        Docstring for forward

        :param inputs: 2D input vector [input_size, batch_size]

        1) apply weights to get coefficients (diff shape for sequential vs. parallel)
        2) pass coefficients into ladder(s)
        3) apply final linear layer to combine ladder results (also true for sequential since keep prev results)
        '''

        if self.is_log:
            C = abs(inputs.min().item()) + 0.0001
            inputs = inputs + C
            inputs = torch.log(inputs) #apply log before weights

        #apply weights: (if sequential, will get coefficients for first ladder only, else will get coefficients for all ladders)
        coeffs = self.coeff_weight(inputs) # [batch, first_depth] or [batch, sum(depth_list)]

        if self.is_seq:
            curr_inputs = coeffs
            ladder_outputs = []
            
            # Pre-calculate all fillers in one fast pass
            all_fillers = self.res_con_up(self.res_con_down(inputs))
            filler_list = torch.split(all_fillers, self.filler_sizes, dim=1)
            
            for i, ladder in enumerate(self.layers):
                # Now curr_inputs will match the ladder's expected dimension
                out = ladder(curr_inputs)
                ladder_outputs.append(out)
                
                if i < len(filler_list):
                    # Concatenate to form the input for the NEXT ladder
                    curr_inputs = torch.cat((out.view(-1, 1), filler_list[i]), dim=1)
            
            output = torch.stack(ladder_outputs, dim=1)
            
        else: #parallel ladders
            
            # split coeffs based on depth_list
            ladder_inputs = torch.split(coeffs, self.depth_list, dim=1)

            # map ladders to their respective input slices
            ladder_outputs = []
            for ladder, current_input in zip(self.layers, ladder_inputs):
                out = ladder(current_input)
                ladder_outputs.append(out)
            
            output = torch.stack(ladder_outputs, dim=1)
    
        if self.is_reciprocal:
            # Apply reciprocal to each ladder output
            output = 1.0 / output

        # Apply final linear layer
        output = self.final_layer(output)

        # Depending on whether this is a regression task, may need different formatting:
        return output.view(-1) if self.output_size == 1 else output
    
class Ladder(nn.Module):
    def __init__(self, variant_settings, depth, dropout):
        '''
        Docstring for __init__

        :param variant_settings: dictionary of settings for model variant
        :param depth: depth of ladder
        :param dropout: dropout rate to prevent overfitting, typical values between 0.2 and 0.5
        '''

        super().__init__()
        
        #unpack variant settings
        self.is_cofr = variant_settings['is_cofr'] #using continued fractions (True) or continued logarithms (False)?
        self.is_bin = variant_settings['is_bin'] #binarizing coefficients before recurrence?
        self.is_orig = variant_settings['is_orig'] #using original formula instead of continuant form?
        self.is_seq = variant_settings['is_seq'] #using sequential ladders instead of parallel ones?
        self.is_euc_dist = variant_settings['is_euc_dist'] #using euclidean distance instead of dot product with weights to get coefficients?
        self.is_reciprocal = variant_settings['is_reciprocal'] #applying reciprocal after each ladder?
        self.is_dist_norm = variant_settings['is_dist_norm'] #normalizing coefficients to be unit vector?
        self.is_log = variant_settings['is_log'] #apply log before weights?

        self.depth = depth
        
        self.norm = nn.LayerNorm(depth) #normalization layer
        self.dropout = nn.Dropout(p=dropout) # optional dropout layer

    def forward(self, inputs):
        '''
        Docstring for forward

        :param inputs: 2D input vector [batch, depth]
        
        1) apply normalization
        2) binarize if CoLogB
        3) apply recurrence formula (original or continuant)
        4) apply dropout

        '''

        # Coefficients: [batch, depth]
        coeffs = self.norm(inputs)

        # Binarize if CoLogB
        if self.is_bin:
            coeffs = binarize_with_ste(coeffs)

        if self.is_dist_norm:
            # normalize coefficients to be unit vector
            l2_norm = torch.linalg.norm(coeffs)
            coeffs = coeffs / l2_norm #normalize coefficients

        # Transpose for recurrence: [depth, batch]
        coeffs_t = coeffs.t()

        # Recurrence --> [batch]
        output = self.recurrence_formula_orig(coeffs_t, self.is_cofr) if self.is_orig else self.recurrence_formula_cont(coeffs_t, self.is_cofr)

        # Apply dropout
        output = self.dropout(output.view(-1, 1)) # ensuring output is one column of length batch_size

        # Return as [batch] for stacking
        return output.view(-1)
    
    @staticmethod
    def recurrence_formula_cont(X: torch.Tensor, is_cofr : bool): 
        # X: tensor of exponents for recurrence terms, 
        # is_cofr - is this a continued fraction or continued logarithm?
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

        if is_cofr:
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
            if is_cofr:
                dep_term = 1.0
            else:
                dep_term = C[i-1]

            A_n = C[i] * A_prev + dep_term * A_prev2 #next numerator term
            B_n = C[i] * B_prev + dep_term * B_prev2 #next denominator term
            A_prev2, A_prev = A_prev, A_n #prepping new A_{n-2} and A{n-1} terms
            B_prev2, B_prev = B_prev, B_n #ditto

        return A_n / (B_n + epsilon) #add small epsilon if cofrnet

    @staticmethod
    def recurrence_formula_orig(X: torch.Tensor, is_cofr : bool):
        # X: tensor of exponents for recurrence terms,
        '''
        Computes value of continued logarithm using original formula:
        C_0 + C_0/(C_1 + C_1/(C_2 + ...)), where C_i = 2^(X[i])
       
        Working backwards, calculates C_{n-1} + C_{n-1}/C_{n}, then C_{n-2} + C_{n-2}/(C_{n-1} + C_{n-1}/C_{n}), ...
        But since order doesn't matter, we actually calculate C_1 + C_1/C_0, then C_2 + C_2/(C_1 + C_1/C_0), ...

        In other words, the formula is:
        A_n = C_n + C_n/A_{n-1}

        Also a COFRNET version, where the formula is instead:
        A_n = C_n + 1/A_{n-1}
        '''

        n = X.shape[0]

        if is_cofr:
            C = X
            epsilon = 0.1
        else:
            C = torch.pow(2, X) #raise 2^x for each value
            epsilon = 0.0

        #base cases:
        if n == 1:
            return C[0]

        A_prev = C[0]

        #recursive step
        for i in range(1, n):
            if is_cofr:
                C_term = 1.0
            else:
                C_term = C[i]

            A_n = C[i] + C_term / (A_prev + epsilon) #next term
            A_prev = A_n #prepping new A_{n-1} term
        
        return A_n