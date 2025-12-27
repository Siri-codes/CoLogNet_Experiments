#Data Processing Utils

import torch
from torch.utils.data import DataLoader
from enum import Enum

import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import Subset

#Data Processing
class Dataset_Enum(Enum):
    '''
    Datasets used for testing:
        - MNIST: multi-class grayscale image classification (10 classes)
        - WAVEFORM: classification (3 classes)
        - BOSTON HOUSING: regression (predict median house price)

    '''
    MNIST = "mnist"
    WAVEFORM = "waveform"
    BOSTON = "boston housing"

def process_data(dataset, data_dir, batch_size):
    '''
    Creates train and test dataloaders for *dataset*
    '''

    y_scaler = None #only scale data for regression task

    # DATA LOADING & PREPROCESSING
    if dataset == Dataset_Enum.MNIST: # MNIST

        # Train transform (includes augmentation)
        # augmentation teaches model to recognize digits even if they are slightly tilted or shifted
        train_transform = v2.Compose([
            v2.ToImage(),                            # convert to image tensor
            v2.ToDtype(torch.float32, scale=True),   # scale to [0, 1]
            v2.RandomRotation(degrees=10),           # tilt digits up to 10 deg
            v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)), #translation, scaling
            v2.Normalize((0.1307,), (0.3081,)),      # Standard MNIST norm (subtract mean, divide by std)
            v2.Lambda(lambda x: x.view(-1))          # Flatten data to 1d vector
        ])

        # Validation/Test transform (no augmentation --- represents actual performance with real data)
        eval_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.1307,), (0.3081,)),
            v2.Lambda(lambda x: x.view(-1))
        ])

        # Instantiate the dataset twice with different transforms for augmented (train) and real (val) versions of dataset
        full_train_augmented = datasets.MNIST(data_dir, train=True, download=True, transform=train_transform)
        full_train_clean = datasets.MNIST(data_dir, train=True, download=True, transform=eval_transform)
        
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=eval_transform)

        # Create shared indices to split the training data (90/10 train-val split)
        indices = torch.randperm(len(full_train_augmented)) #shuffle indices: random permutation of ints [0, n)
        val_size = int(0.1 * len(full_train_augmented)) #10% used for validation set, 90% used for train set
        train_indices = indices[val_size:] #last 90% used for training
        val_indices = indices[:val_size] #first 10% used for validation

        # Create datasets using shared indices and different (augmented/real) versions of data
        train_dataset = Subset(full_train_augmented, train_indices)
        val_dataset = Subset(full_train_clean, val_indices)

    else:
        # Tabular data (Boston Housing & Waveform)
        if dataset == Dataset_Enum.BOSTON: # Boston
            df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/refs/heads/master/BostonHousing.csv') #collect data
            target_col, is_regression = 'medv', True 
        else: # Waveform
            df = pd.read_csv('http://www.dropbox.com/s/qtdv1teptf097zl/waveformnoise.csv?dl=1')
            target_col, is_regression = df.columns[-1], False

        # Split input features/target
        X = df.drop(target_col, axis=1).values
        y = df[target_col].values

        # Split before scaling to avoid leakage (i.e. don't want test data in the training data)
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2) #train/test
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2)  #train/val

        # Normalize features so all contribute equally to result
        scaler = StandardScaler() # standardization/Z score normalization: every pt has mean of 0, std of 1
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        if is_regression: #if regression task, need to scale targets to reasonable ranges
            y_scaler = StandardScaler()

            # Scale train, val, and test labels
            # .reshape(-1, 1) because the scaler expects a 2D array
            y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_val = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
            y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

        def to_tensor_ds(X_data, y_data, regression):  # helper: convert to TensorDataset
            return TensorDataset(
                torch.tensor(X_data, dtype=torch.float32),
                torch.tensor(y_data, dtype=torch.float32 if regression else torch.long) #ints for classification
            )

        train_dataset = to_tensor_ds(X_train, y_train, is_regression)
        val_dataset = to_tensor_ds(X_val, y_val, is_regression)
        test_dataset = to_tensor_ds(X_test, y_test, is_regression)

    # DATALOADER CREATION
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, y_scaler

#Plotting Utils
def plot_loss_curves(history):
    '''
    Docstring for plot_loss_curves
    

    :param history: dictionary containing training history with keys 'train_loss' and 'val_loss'
    '''
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
