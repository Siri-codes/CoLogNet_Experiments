#TRAIN/TEST LOOPS

import torch # import main library
from sklearn.metrics import r2_score

import wandb
from math import floor

from CoLogNet_Experiments.utils.data_processing import Dataset_Enum, process_data, plot_loss_curves
from CoLogNet_Experiments.models.colognet import ContNet_Model, Variant
from CoLogNet_Experiments.models.mlp import MLP
from CoLogNet_Experiments.models.mlp import SwiGLUMLP
from CoLogNet_Experiments.utils.train import train


def train_test_loop(dataset_enum, model_type, depths, learning_rate, dropout, batch_size, num_epochs, weight_decay=1e-4, num_hidden=1):
  '''
  Docstring for train_test_loop
  Customizeable train/test loop: Trains and evaluates a model on the specified dataset with given hyperparameters.
  
  :param dataset: the dataset to use (not necesarily one of the standard enums)
  :param model_type: the type of model to use (e.g., Variant.MLP)
  :param depths: list specifying the depth of each layer in the model
  :param learning_rate: learning rate for the optimizer
  :param dropout: dropout rate 
  :param batch_size: batch size for training
  :param num_epochs: number of epochs to train
  :param weight_decay: weight decay (L2 regularization) for the optimizer
  '''

  #get dataset (train, test, val, specifies which dataset it is)
  train_loader, val_loader, test_loader, y_scaler = process_data(dataset_enum, batch_size)

  input_size = dataset_enum.input_size
  output_size = dataset_enum.output_size
  is_regression = dataset_enum.is_regression

  model = get_model(model_type, input_size, output_size, depths, dropout, num_hidden)

  #train the model
  history = train(model, train_loader, val_loader, is_regression, num_epochs=num_epochs, lr=learning_rate, weight_decay=weight_decay)
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

        input_size = dataset.input_size
        output_size = dataset.output_size

        base_depth = config.base_depth
        num_ladders = config.num_ladders

        # 3 different depth configurations: Uniform, Even_Odd, Pyramid
        #ex: if base_depth = 2 and num_ladders = 4
        if config.config_type == "Uniform": # [2, 2, 2, 2]
            depths = [config.base_depth] * config.num_ladders
        elif config.config_type == "Even_Odd": # [2, 3, 2, 3]
            depths = [config.base_depth if i % 2 == 0 else config.base_depth + 1 for i in range(config.num_ladders)]
        elif config.config_type == "Pyramid": # [2, 4, 6, 8] 
            depths = [config.base_depth + (i * 2) for i in range(config.num_ladders)]
        
        result_metric, total_params = train_test_loop(dataset, model_type, depths, config.lr, config.dropout, config.batch_size, num_epochs=50, weight_decay=1e-4, num_hidden=config.num_hidden)
           
        wandb.log({"score": result_metric, "total_params": total_params})


def train_test_wandb_lim_params():
    '''
    Docstring for train_test_wandb
    Sets up and runs a training/testing loop with hyperparameters and accuracy logged to Weights & Biases (W&B).
    '''

    # Initialize a W&B run
    with wandb.init() as run:
        config = run.config # Retrieve hyperparameters from W&B config
        
        model_type = Variant[config.model_type]

        dataset = Dataset_Enum[config.dataset]

        input_size = dataset.input_size
        output_size = dataset.output_size

        max_params = config.max_params
        num_ladders = config.num_ladders

        if config.config_type == "Uniform":
            # subtract fixed combiner params first
            remaining = max_params - (output_size * num_ladders + output_size)
            # Solve for d: remaining = num_ladders * d * (input_size + 1)
            base_depth = floor(remaining / (num_ladders * (input_size + 1)))
            depths = [max(1, base_depth)] * num_ladders
        elif config.config_type == "Even_Odd":
            remaining_budget = max_params - (output_size * num_ladders + output_size)
            # Total depth sum across all ladders if base_depth is d
            # total_d = (base_depth * num_ladders) + (num_ladders // 2)
            # Solve for base_depth:
            base_depth = floor((remaining_budget / (input_size + 1) - (num_ladders // 2)) / num_ladders)
            depths = [max(1, base_depth if i % 2 == 0 else base_depth + 1) for i in range(num_ladders)]
        elif config.config_type == "Pyramid":
            remaining_budget = max_params - (output_size * num_ladders + output_size)
            sum_increments = num_ladders * (num_ladders - 1)
            # Solve for d: remaining_budget = (num_ladders * d + sum_increments) * (input_size + 1)
            base_depth = floor((remaining_budget / (input_size + 1) - sum_increments) / num_ladders)
            depths = [max(1, base_depth + (i * 2)) for i in range(num_ladders)]

        '''
        # 3 different depth configurations: Uniform, Even_Odd, Pyramid
        #ex: if base_depth = 2 and num_ladders = 4
        if config.config_type == "Uniform": # [2, 2, 2, 2]
            depths = [config.base_depth] * config.num_ladders
        elif config.config_type == "Even_Odd": # [2, 3, 2, 3]
            depths = [config.base_depth if i % 2 == 0 else config.base_depth + 1 
                      for i in range(config.num_ladders)]
        elif config.config_type == "Pyramid": # [2, 4, 6, 8] 
            depths = [config.base_depth + (i * 2) for i in range(config.num_ladders)]
        '''

        result_metric, total_params = train_test_loop(dataset, model_type, depths, config.lr, config.dropout, config.batch_size, num_epochs=50, weight_decay=1e-4, num_hidden=config.num_hidden)
           
        wandb.log({"score": result_metric, "total_params": total_params})

def get_model(model_type, input_size, output_size, depths, dropout, num_hidden):
    '''
    Get a model based on the specified model type and parameters.
    :param model_type: the type of model to create
    :param input_size: input size of the model
    :param output_size: output size of the model
    :param depths: list of ladder depths
    '''
    
    if model_type is Variant.MLP:
        # Calculate target parameters based on ContNet architecture
        target_params = calculate_parameters(input_size, output_size, depths)
        model = MLP(input_size, output_size, target_params, dropout, num_hidden)
    elif model_type is Variant.SWIGLU:
        target_params = calculate_parameters(input_size, output_size, depths)
        model = SwiGLUMLP(input_size, output_size, depths, dropout)
    else:
        model = ContNet_Model(model_type, input_size, output_size, depths, dropout)
    return model

def eval_model(model, test_loader, is_regression, y_scaler): 

    '''
    Docstring for eval_model
    Evaluates the model (accuracy for classification, R2 score for regression) on the provided dataloader.
    
    :param dataset: the dataset to evaluate on (includes test set + y_scaler)
    :param model: the model to evaluate

    '''

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

        #print(f'PREDS: {preds}')
        #print(f'TARGETS: {targets}')

        # Return R2 Score --- aiming for close to 1, with 0.7-0.9 considered a strong score
        return r2_score(targets, preds)
    else:
        return num_correct / num_total # Return accuracy (0 to 1)


@staticmethod
def calculate_parameters(input_size, output_size, depths):
  '''
  Calculates the total number of parameters for a ContNet model given input size, output size, and ladder depths.

  :param input_size: input size of the model
  :param output_size: output size of the model
  :param depths: list of ladder depths
  '''

  # 1. Calculate parameters for all ladders
  # Each ladder 'l_i' has a weight matrix of (input_size * depth_i) and bias of (depth_i)
  ladder_params_total = sum(d * input_size + d for d in depths)

  # 2. Calculate parameters for the final combiner
  # weight matrix: (output_size * sum(depths)), bias: (output_size)
  combiner_params = output_size * len(depths) + output_size

  # 3. Total Model Parameters
  target_params = ladder_params_total + combiner_params

  return target_params
