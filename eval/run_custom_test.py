#CUSTOM TEST SCRIPT

from utils.data_processing import Dataset_Enum
from eval.colognet_test_loops import train_test_loop
from models.colognet import Variant

# Choose dataset, model_type, hyperparameters:
dataset = Dataset_Enum.MNIST
model_type = Variant.COLOGNET
depths = [4, 4, 4, 4, 4]
lr = 0.0005
dropout = 0.1
batch_size = 200
num_epochs = 30
weight_decay = 1e-4

# Run train/test loop with specified parameters
# The train/test loop will print out/plot results
train_test_loop(dataset, model_type, depths, lr, dropout, batch_size, num_epochs, weight_decay)