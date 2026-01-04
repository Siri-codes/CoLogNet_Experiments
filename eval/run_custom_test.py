"""
Custom Test Script

To run the following code via terminal: 
    1) navigate to the root directory (CoLogNet_Experiments)
    2) run: python -m eval.run_custom_test
    
"""

import __main__
from CoLogNet_Experiments.utils.data_processing import Dataset_Enum
from CoLogNet_Experiments.eval.colognet_test_loops import train_test_loop
from CoLogNet_Experiments.models.colognet import Variant

# Choose dataset, model_type, hyperparameters:
dataset = Dataset_Enum.BOSTON
model_type = Model_Type.CONTNET

variant_settings = {
    'is_cofr' : False,
    'is_bin' : False,
    'is_orig' : False,
    'is_seq' : False,
    'is_euc_dist' : False,
    'is_reciprocal' : False,
    'is_dist_norm' : False,
    'is_log' : False
}
depths = [4, 4, 4, 4]
lr = 0.05
dropout = 0.0
batch_size = 128
num_epochs = 50
weight_decay = 1e-4
num_hidden = 1

def main():
    '''
    Run train/test loop with specified parameters
    '''

    # The train/test loop will print out/plot results
    train_test_loop(dataset, model_type, variant_settings, depths, lr, dropout, batch_size, num_epochs, weight_decay, num_hidden)

if __name__ == "__main__":
    main()