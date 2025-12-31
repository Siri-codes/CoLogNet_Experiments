#Train Function
import torch
import torch.nn as nn
import torch.optim as optim

from CoLogNet_Experiments.utils.data_processing import Dataset_Enum
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

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

def train_general(train_loader, val_loader, is_regression, model, num_epochs, lr, weight_decay):
    '''
    Trains a model using provided 'dataloaders'
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

        for inputs, targets in train_loader:
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

def train_mnist(train_dataset, val_dataset, is_regression, model, num_epochs, lr, weight_decay, batch_size):
    '''
    Trains a model on the MNIST dataset without using a dataloader because that makes it take a long time
    '''

    # inputs and targets prepared as tensors
    train_inputs = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    train_targets = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])

    val_inputs = torch.stack([val_dataset[i][0] for i in range(len(val_dataset))])
    val_targets = torch.tensor([val_dataset[i][1] for i in range(len(val_dataset))])

    # Create the Fast Loader objects
    train_loader = FastTensorLoader(train_inputs, train_targets, batch_size=batch_size, shuffle=True)
    val_loader = FastTensorLoader(val_inputs, val_targets, batch_size=batch_size, shuffle=False)
        
    return train_general(train_loader, val_loader, is_regression, model, num_epochs, lr, weight_decay)

def train_dataloader(train_dataset, val_dataset, model, num_epochs, lr, weight_decay, batch_size):
    '''
    Trains a model using dataloader
    '''
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_general(train_loader, val_loader, model, num_epochs, lr, weight_decay)

def train(dataset, model, num_epochs, lr, weight_decay):

    '''
    Trains a model on the given dataset.
    '''

    dataset_enum = dataset.dataset_enum
    batch_size = dataset.batch_size

    if dataset_enum is Dataset_Enum.MNIST:
        return train_mnist(dataset.train_dataset, dataset.val_dataset, False, model, num_epochs, lr, weight_decay, batch_size)
    elif dataset_enum is Dataset_Enum.WAVEFORM:
        return train_dataloader(dataset.train_dataset, dataset.val_dataset, False, model, num_epochs, lr, weight_decay, batch_size)
    elif dataset_enum is Dataset_Enum.BOSTON:
        return train_dataloader(dataset.train_dataset, dataset.val_dataset, True, model, num_epochs, lr, weight_decay, batch_size)

def eval_general(test_loader, is_regression, model, y_scaler): #helper for eval_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() # Set model to evaluation mode

    num_correct = 0
    num_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            if is_regression:
                # Store values to calculate R2 or MSE later
                all_preds.append(outputs.cpu())
                all_labels.append(labels.cpu())
            else:
                # Classification logic
                _, predicted = torch.max(outputs, 1)
                num_correct += (predicted == labels).sum().item()
                num_total += labels.size(0)

    if is_regression:
        preds = torch.cat(all_preds).numpy()
        targets = torch.cat(all_labels).numpy()

        # Convert back to original scale if scaler is provided
        if y_scaler is not None:
            preds = y_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
            targets = y_scaler.inverse_transform(targets.reshape(-1, 1)).flatten()

        print(f'PREDS: {preds}')
        print(f'TARGETS: {targets}')

        # Return R2 Score --- aiming for close to 1, with 0.7-0.9 considered a strong score
        return r2_score(targets, preds)
    else:
        return num_correct / num_total # Return accuracy (0 to 1)

def eval_model(dataset, model):
  '''
  Docstring for eval_model
  Evaluates the model (accuracy for classification, R2 score for regression) on the provided dataloader.
  
  :param dataset: the dataset to evaluate on (includes test set + y_scaler)
  :param model: the model to evaluate

  '''

  dataset_enum = dataset.dataset_enum
  batch_size = dataset.batch_size

  if dataset_enum is Dataset_Enum.MNIST:
    test_inputs = torch.stack([dataset.test_dataset[i][0] for i in range(len(dataset.test_dataset))])
    test_targets = torch.tensor([dataset.test_dataset[i][1] for i in range(len(dataset.test_dataset))])
    test_loader = FastTensorLoader(test_inputs, test_targets, batch_size=batch_size, shuffle=False)
  else:
    test_loader = Dataloader(dataset.test_dataset, batch_size=batch_size, shuffle=False)
    
  is_regression = dataset_enum.is_regression
  y_scaler = dataset.y_scaler

  return eval_general(test_loader, is_regression, model, y_scaler)

