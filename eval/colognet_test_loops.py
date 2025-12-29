#TRAIN/TEST LOOPS

import torch # import main library
from sklearn.metrics import r2_score

import wandb

from CoLogNet_Experiments.utils.data_processing import Dataset_Enum, process_data, plot_loss_curves
from CoLogNet_Experiments.models.colognet import ContNet_Model, Variant
from CoLogNet_Experiments.models.mlp import MLP
from CoLogNet_Experiments.models.mlp import SwiGLUMLP
from CoLogNet_Experiments.utils.train import train


def train_test_loop(dataset, model_type, depths, learning_rate, dropout, batch_size, num_epochs, weight_decay=1e-4):
  '''
  Docstring for train_test_loop
  Customizeable train/test loop: Trains and evaluates a model on the specified dataset with given hyperparameters.
  
  :param dataset: the dataset to use (e.g., Dataset_Enum.MNIST)
  :param model_type: the type of model to use (e.g., Variant.MLP)
  :param depths: list specifying the depth of each layer in the model
  :param learning_rate: learning rate for the optimizer
  :param dropout: dropout rate 
  :param batch_size: batch size for training
  :param num_epochs: number of epochs to train
  :param weight_decay: weight decay (L2 regularization) for the optimizer
  '''

  #get data loaders
  train_loader, val_loader, test_loader, y_scaler = process_data(dataset, data_dir="./data", batch_size=batch_size)

  if dataset is Dataset_Enum.MNIST:
    input_size = 784
    output_size = 10
    is_regression = False
  elif dataset is Dataset_Enum.WAVEFORM:
    input_size = 40
    output_size = 3
    is_regression = False
  elif dataset is Dataset_Enum.BOSTON:
    input_size = 13
    output_size = 1
    is_regression = True

  if model_type is Variant.MLP:
    model = MLP(input_size, output_size, depths, dropout)
  elif model_type is Variant.SWIGLU:
    model = SwiGLUMLP(input_size, output_size, depths, dropout)
  else:
    model = ContNet_Model(model_type, input_size, output_size, depths, dropout)

  #train the model
  history = train(model, train_loader, val_loader, num_epochs=num_epochs, lr=learning_rate, weight_decay=weight_decay, is_regression=is_regression)
  plot_loss_curves(history) #optional: plot loss curves

  total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) 

  print("Total trainable parameters:", total_params) # optional: print total parameters

  # Evaluate on test set
  result_metric = eval_model(model, test_loader, is_regression, y_scaler)

  if is_regression:
      print(f"R2 Score: {result_metric}")
  else:
      print(f"Accuracy: {result_metric}")

  return result_metric, total_params

def train_test_wandb():
    '''
    Docstring for train_test_wandb
    Sets up and runs a training/testing loop with hyperparameters and accuracy logged to Weights & Biases (W&B).
    '''

    # Initialize a W&B run
    with wandb.init() as run:
        config = run.config # Retrieve hyperparameters from W&B config
        
        model_type = Variant[config.model_type]

        dataset = Dataset_Enum[config.dataset]
        
        # 3 different depth configurations: Uniform, Even_Odd, Pyramid
        #ex: if bbase_depth = 2 and num_ladders = 4
        if config.config_type == "Uniform": # [2, 2, 2, 2]
            depths = [config.base_depth] * config.num_ladders
        elif config.config_type == "Even_Odd": # [2, 3, 2, 3]
            depths = [config.base_depth if i % 2 == 0 else config.base_depth + 1 
                      for i in range(config.num_ladders)]
        elif config.config_type == "Pyramid": # [2, 4, 6, 8]
            depths = [config.base_depth + (i * 2) for i in range(config.num_ladders)]

        result_metric, total_params = train_test_loop(dataset, model_type, depths, config.lr, config.dropout, config.batch_size, num_epochs=50, weight_decay=1e-4)
           
        wandb.log({"score": result_metric, "total_params": total_params})

def eval_model(model, dataloader, is_regression=False, y_scaler=None):
  '''
  Docstring for eval_model
  Evaluates the model (accuracy for classification, R2 score for regression) on the provided dataloader.
  
  :param model: the model to evaluate
  :param dataloader: the dataloader containing the evaluation data
  :param is_regression: Boolean indicating if the task is regression (True) or classification (False)
  :param y_scaler: the scaler used to inverse transform the target values for regression tasks
  '''
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  model.eval() # Set model to evaluation mode

  num_correct = 0
  num_total = 0
  all_preds = []
  all_labels = []

  with torch.no_grad():
    for inputs, labels in dataloader:
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
