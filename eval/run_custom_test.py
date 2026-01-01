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
dataset = Dataset_Enum.MNIST #options: MNIST, CIFAR10, WAVEFORM, BOSTON
model_type = Variant.COLOGNET #options: MLP, SWIGLU, COFRNET, COLOGNET, COLOGNET_B, COLOGNET_E
depths = [4, 4, 4, 4, 4] 
lr = 0.0005
dropout = 0.1 #recommended range: [0.0 - 0.5]
batch_size = 200
num_epochs = 30
weight_decay = 1e-4
num_hidden = 1 #only for MLP -- number of hidden layers

def main():
    '''
    Run train/test loop with specified parameters
    '''

    # The train/test loop will print out/plot results
    train_test_loop(dataset, model_type, depths, lr, dropout, batch_size, num_epochs, weight_decay)

if __name__ == "__main__":
    main()