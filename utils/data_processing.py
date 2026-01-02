#Data Processing Utils

import torch
from enum import Enum

import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import Subset
from torch.utils.data import DataLoader

#Data Processing

class Dataset_Enum(Enum):
    '''
    Datasets used for testing:
        - MNIST: multi-class grayscale image classification (10 classes)
        - WAVEFORM: classification (3 classes)
        - BOSTON HOUSING: regression (predict median house price)
        - CIFAR10: multi-class color image classification (10 classes)
    '''
    
    # (name, input size, output size, is_regression, standard_dataloader)
    # is_regression: is this a regression task?

    MNIST = ("mnist", 784, 10, False) 
    CIFAR10 = ("cifar10", 3072, 10, False) 
    WAVEFORM = ("waveform", 40, 3, False)
    BOSTON = ("boston housing", 13, 1, True)
 
    def __init__(self, value, input_size, output_size, is_regression):
        self.input_size = input_size
        self.output_size = output_size
        self.is_regression = is_regression

class FastTensorLoader:
    '''
    DataLoader-like object for a set of tensors.
    In case of MNIST dataset, faster than DataLoader.

    '''
    def __init__(self, inputs, targets, batch_size=64, shuffle=True):
        # Select device here to ensure tensors are moved immediately
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inputs = inputs.to(self.device)
        self.targets = targets.to(self.device)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = self.inputs.size(0)

    def __iter__(self):
        indices = torch.arange(self.num_samples, device=self.device)
        if self.shuffle:
            indices = indices[torch.randperm(self.num_samples)]
        
        for i in range(0, self.num_samples, self.batch_size):
            batch_idx = indices[i : i + self.batch_size]
            yield self.inputs[batch_idx], self.targets[batch_idx]

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

def to_tensor_dataloader(dataset, batch_size):
    # inputs and targets prepared as tensors
    inputs = torch.stack([dataset[i][0] for i in range(len(dataset))])
    targets = torch.tensor([dataset[i][1] for i in range(len(dataset))])

    # Create the Fast Loader objects
    fast_loader = FastTensorLoader(inputs, targets, batch_size=batch_size, shuffle=True)
    
    return fast_loader


#DATA LOADING AND PROCESSING:
def process_data(dataset_enum, batch_size):
    '''
    Returns dataset with proper train, test, and val datasets to be used for Dataloaders (Waveform, Boston) or directly for training (MNIST)
    Also has y_scaler for regression tasks (Boston)

    '''
    if dataset_enum is Dataset_Enum.MNIST: # MNIST
        train_loader, val_loader, test_loader, y_scaler = process_data_mnist(batch_size)
    elif dataset_enum is Dataset_Enum.CIFAR10:
        train_loader, val_loader, test_loader, y_scaler = process_data_cifar10(batch_size)
    elif dataset_enum is Dataset_Enum.WAVEFORM:
        train_loader, val_loader, test_loader, y_scaler  = process_data_waveform(batch_size)
    elif dataset_enum is Dataset_Enum.BOSTON:
        train_loader, val_loader, test_loader, y_scaler  = process_data_boston(batch_size)
    
    return train_loader, val_loader, test_loader, y_scaler

def process_data_mnist(batch_size, data_dir='/tmp/data'):

    # Simple version without augmentation for simplicity/speed
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.1307,), (0.3081,)),
        v2.Lambda(lambda x: torch.flatten(x))
    ])

    # Load datasets
    full_train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    return process_data_custom_dataloader(full_train, test_dataset, batch_size)

def process_data_cifar10(batch_size, data_dir='/tmp/data'):
    
    # CIFAR10 normalization stats
    means = (0.4914, 0.4822, 0.4465)
    stds = (0.2470, 0.2435, 0.2616)

    # standard transform (no augmentation)
    transform = v2.Compose([
        v2.ToImage(),                            # Convert to tensor image
        v2.RandomHorizontalFlip(p=0.5), #DATA AUGMENTATION
        v2.RandomCrop(32, padding=4),   #DATA AUGMENTATION
        v2.ToDtype(torch.float32, scale=True),   # Convert to float and scale to [0, 1]
        v2.Normalize(means, stds),
        v2.Lambda(lambda x: torch.flatten(x))
    ])

    # Load the full train and test datasets
    full_train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    return process_data_custom_dataloader(full_train_dataset, test_dataset, batch_size)


def process_data_custom_dataloader(full_train, test_dataset, batch_size, val_split=0.1, data_dir='/tmp/data'):
    '''
    Docstring for process_data_custom_dataloader
    Split data into train, val, and test sets
    Create custom dataloaders for each set (not pytorch Dataloader bc slower for MNIST and CIFAR10)
    '''

    # Stratified train/val split 
    train_idx, val_idx = train_test_split(
        range(len(full_train)),
        test_size=val_split,
        stratify=full_train.targets, # Keeps 10% of EACH digit for validation
        random_state=42
    )

    train_dataset = Subset(full_train, train_idx)
    val_dataset = Subset(full_train, val_idx)

    train_loader = to_tensor_dataloader(train_dataset, batch_size)
    val_loader = to_tensor_dataloader(val_dataset, batch_size)
    test_loader = to_tensor_dataloader(test_dataset, batch_size)

    return train_loader, val_loader, test_loader, None

def process_data_waveform(data_dir='/tmp/data'):
    df = pd.read_csv('http://www.dropbox.com/s/qtdv1teptf097zl/waveformnoise.csv?dl=1')
    target_col = df.columns[-1]

    return tabular_data_helper(df, target_col, False)


def process_data_boston(data_dir='/tmp/data'):
    df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/refs/heads/master/BostonHousing.csv') #collect data
    target_col = 'medv'

    return tabular_data_helper(df, target_col, True)

def tabular_data_helper(df, target_col, is_regression):
    '''
    Docstring for tabular_data_helper
    split data into train, val, and test sets
    normalize features (if is_regression, normalize both inputs and outpus, else just inputs)
    return standard pytorch dataloaders for each set

    :param df: dataframe containing data
    :param target_col: name of target column
    :param is_regression: is this a regression task (requires output scaling)
    '''
    # Split input features/target
    X = df.drop(target_col, axis=1).values
    y = df[target_col].values

    # Split before scaling to avoid leakage (i.e. don't want test data in the training data)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2) #train/test
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2)  #train/val

    # Normalize input features so all contribute equally to result
    scaler = StandardScaler() # standardization/Z score normalization: every pt has mean of 0, std of 1
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    if is_regression:
        y_scaler = StandardScaler()

        # Scale train, val, and test labels (since regression task)
        # .reshape(-1, 1) because the scaler expects a 2D array
        y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
        y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
    else:
        y_scaler = None

    train_dataset = to_tensor_ds(X_train, y_train, False)
    val_dataset = to_tensor_ds(X_val, y_val, False)
    test_dataset = to_tensor_ds(X_test, y_test, False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader, y_scaler


def to_tensor_ds(X_data, y_data, regression):  
    # helper: convert to TensorDataset
    return TensorDataset(
        torch.tensor(X_data, dtype=torch.float32),
        torch.tensor(y_data, dtype=torch.float32 if regression else torch.long) #ints for classification
    )
