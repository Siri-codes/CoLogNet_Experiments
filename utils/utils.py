#Data Processing Utils

from xml.parsers.expat import model
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from enum import Enum

import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms

from torchvision.transforms import v2
from torch.utils.data import Dataset
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

#Train Function
def train(model, dataloader, val_loader, num_epochs, lr, weight_decay, is_regression):

    '''
    Trains a model on the given dataset.
    '''

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_fn = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) #could change optimizer if needed

    # Dynamic LR scheduling for better convergence (reduces learning rate over time)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    #for early stopping:
    best_loss = float('inf')
    patience, counter = 5, 0

    # Initialize training history dictionary --- useful for plotting loss curves
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],}

    for epoch in range(num_epochs):

        #TRAINING
        model.train() # set to training mode

        train_running_loss = 0.0 #total loss over batches
        train_running_correct = 0 # For accuracy (classification)
        train_running_mae = 0.0   # For mean absolute error (regression)
        total_samples = 0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to same device

            #Calculate loss by comparing predictions (outputs) vs. targets:
            outputs = model(inputs)

            # Regression targets need to be reshaped to (batch, 1)
            if is_regression:
                    loss = loss_fn(outputs.view(-1), targets.view(-1))
                    train_running_mae += torch.abs(outputs - targets).sum().item()
            else:
                    loss = loss_fn(outputs, targets)
                    _, predicted = torch.max(outputs, 1) # Get predicted class index
                    train_running_correct += (predicted == targets).sum().item()

            #backward pass/gradient descent:
            optimizer.zero_grad(set_to_none=True) #reset grads
            loss.backward() #calculate new grads
            optimizer.step() #update params

            train_running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            # Average metrics for the epoch
            epoch_train_loss = train_running_loss / total_samples
            epoch_train_accuracy = (train_running_correct / total_samples) if not is_regression else (train_running_mae / total_samples)


        # VALIDATION
        model.eval() #set to eval mode

        val_running_loss = 0.0 #total loss over batches
        val_running_correct = 0 # For accuracy
        val_correct = 0
        val_total_samples = 0

        with torch.no_grad(): # (saves memory + faster)
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)  # Move data to same device

                outputs = model(inputs)

                #print(outputs)

                if is_regression:
                    # Ensure both are 1D and scaled
                    # Loss will now be much smaller (e.g., 0.5 instead of 625)
                    loss = loss_fn(outputs.view(-1), targets.view(-1))

                loss = loss_fn(outputs, targets)
                val_running_loss += loss.item()

                # If classification, track accuracy
                if not is_regression:
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == targets).sum().item()
                    val_total_samples += targets.size(0)

        # Early stopping based on validation loss
        if val_running_loss < best_loss:
            best_loss = val_running_loss
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        #UPDATE HISTORY with loss and other metrics
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_accuracy)

        avg_val_loss = val_running_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        if not is_regression:
            avg_val_acc = val_correct / val_total_samples
            history['val_acc'].append(avg_val_acc)

        print(f"Epoch {epoch}: Val Loss: {avg_val_loss:.4f}")

    return history

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