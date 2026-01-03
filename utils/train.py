#Train Function
import torch
import torch.nn as nn
import torch.optim as optim

from CoLogNet_Experiments.utils.data_processing import Dataset_Enum
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score


def train(model, train_loader, val_loader, is_regression, num_epochs, lr, weight_decay, logger=None):
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

        #log if appropriate:
        if logger is not None:
            logger.log("train_loss", epoch_train_loss)
            logger.log("train_acc", epoch_train_accuracy)

        avg_val_loss = val_running_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        if logger is not None:
            logger.log("val_loss", avg_val_loss)

        if not is_regression:
            avg_val_acc = val_correct / val_total_samples
            history['val_acc'].append(avg_val_acc)
            if logger is not None:
                logger.log("val_acc", avg_val_acc)

        print(f"Epoch {epoch}: Train Loss: {epoch_train_loss}; Val Loss: {avg_val_loss:.4f}")

    return history
