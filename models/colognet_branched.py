
import torch # import main library
import torch.nn as nn # import modules
from torch.autograd import Function # import Function to create custom activations
from torch.nn.parameter import Parameter # import Parameter to create custom activations with learnable parameters
import torch.nn.functional as F # import torch functions

import math


class ContNet_Branch_Model(nn.Module):
    '''
    Continued fraction/logarithm model, with branches.

    Rather than the typical two-branch continued fraction:
    A_0 + 1/(A_1 + 1/(A_2 + ...)))

    We can have an arbitrary number of branches:
    layer 1: A_0 + 1/(A_1 + ...) + 1/(B_2 + ...) + 1/(C_1 + ...)

    And each branch has more branches underneath it:
    layer 2: 1/(A_1 + ...) = 1/(A_1 + 1/(A_2 + ...) + 1/(A_3 + ...) + 1/(A_4 + ...)))
    layer 3: 1/(A_2 + ...) = 1/(A_2 + 1/(A_5 + ...) + 1/(A_6 + ...) + 1/(A_7 + ...)) etc.

    '''
    
    def __init__(self, variant_settings: dict, input_size: int, output_size: int, num_ladders : int, branch_sizes: list, dropout: float):
        '''
        Docstring for __init__
        
        :param variant_settings: dictionary of settings for model variant
        :param input_size: size of input data
        :param output_size: size of output vector
        :param num_ladders: number of ladders in model
        :param branch_sizes: list of sizes of branches, also informs how deep branches go (length of list)
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

        self.input_size = input_size
        self.output_size = output_size
        self.num_ladders = num_ladders
        self.layers = nn.ModuleList()

        for i in range(num_ladders):
            l_i = Ladder(variant_settings, branch_sizes, dropout)
            self.layers.append(l_i)

        self.branch_sizes = branch_sizes
        total_coeffs = 1
        for branch_size in branch_sizes:
            total_coeffs = total_coeffs * (1 + branch_size)
        self.total_coeffs = total_coeffs # total coefficients needed per ladder
        
        #weights to get coefficients for recurrence formula
        if self.is_seq:
            self.coeff_weight = nn.Linear(input_size, self.total_coeffs) 
        else:
            self.coeff_weight = Euclidean_Distance_Layer(input_size, self.total_coeffs * num_ladders) if self.is_euc_dist else nn.Linear(input_size, self.total_coeffs * num_ladders)

        if self.is_seq:
            # Calculate the total number of filler terms needed across all ladders
            # skip first ladder, each ladder needs filler_size d_i - 1
            self.filler_size = self.total_coeffs - 1
            self.total_filler_dim = self.filler_size * (num_ladders - 1)
            
            # To save on parameters:
            rank = 16  # Hyperparameter: much smaller than input_size (784)
            self.res_con_down = nn.Linear(input_size, rank, bias=False)
            self.res_con_up = nn.Linear(rank, self.total_filler_dim, bias=True)

            # One big layer instead of many small ones
            self.res_con_net = nn.Linear(input_size, self.total_filler_dim)

        self.final_layer = nn.Linear(in_features=num_ladders, out_features=output_size, bias=True)

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
        coeffs = self.coeff_weight(inputs) 

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
            
            # split coeffs based on num_ladders
            ladder_inputs = torch.chunk(coeffs, self.num_ladders, dim=1)

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
    def __init__(self, variant_settings, branch_sizes, dropout):
        '''
        Docstring for __init__

        :param variant_settings: dictionary of settings for model variant
        :param branch_sizes: list of sizes of branches, also informs how deep branches go (length of list)
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

        self.branch_sizes = branch_sizes
        total_coeffs = 1
        for branch_size in branch_sizes:
            total_coeffs = total_coeffs * (1 + branch_size)
        self.total_coeffs = total_coeffs # total coefficients needed per ladder
        
        self.norm = nn.LayerNorm(self.total_coeffs) #normalization layer
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
        output = self.recurrence_formula(coeffs_t, self.branch_sizes, self.is_cofr) 

        # Apply dropout
        output = self.dropout(output.view(-1, 1)) # ensuring output is one column of length batch_size

        # Return as [batch] for stacking
        return output.view(-1)
    
    
    @staticmethod
    def recurrence_formula(X: torch.Tensor, branch_sizes: list, is_cofr : bool):
        # X: tensor of exponents for recurrence terms,
        '''
        Computes value of branched continued logarithm or continued fraction
        '''

        depth = len(branch_sizes)
        n = X.shape[0]

        if is_cofr:
            C = X
            epsilon = 0.1
        else:
            C = torch.pow(2, X) #raise 2^x for each value
            epsilon = 0.0

        total_sum = C[0] #start with a_0
        idx = 1
        for branch_size in branch_sizes: #for each layer
            layer_sum = C[idx] #start with a_0 term
            
            if is_cofr: #decide numerator term based on whether continued fraction or continued logarithm
                num_term = 1.0
            else:
                num_term = C[idx]
    
            for branch in range(branch_size - 1):
                idx = idx + 1 #increment idx
                
                # add the reciprocal of the branch element
                layer_sum = layer_sum + num_term/C[idx]
                
            A_rec = 1/layer_sum
            total_sum = total_sum + A_rec
            
            # Move idx to the next 'parent' node for the next layer
            idx = idx + 1 
                
        return total_sum
