"""
Train/test loops

Includes:
- A general train_test_loop for customized hyperparameter testing
- A Weights & Biases train_test_loop for hyperparameter optimization (one version with a parameter limit)

"""

import torch # import main library
from sklearn.metrics import r2_score

import wandb
from math import floor
from enum import Enum

from CoLogNet_Experiments.utils.data_processing import Dataset_Enum, process_data
from CoLogNet_Experiments.utils.plotting import plot_loss_curves
from CoLogNet_Experiments.utils.train import train
from CoLogNet_Experiments.models.colognet import ContNet_Model
from CoLogNet_Experiments.models.mlp import MLP
from CoLogNet_Experiments.models.mlp import SwiGLUMLP

class Model_Type(Enum):
    MLP = "mlp"
    SWIGLU = "swiglu"
    CONTNET = "contnet"

def train_test_loop(dataset_enum, model_type, variant_settings, depths, learning_rate, dropout, batch_size, num_epochs, weight_decay=1e-4, num_hidden=1, logger=None):
  train_loader, val_loader, test_loader, y_scaler = process_data(dataset_enum, batch_size)
  input_size = dataset_enum.input_size
  output_size = dataset_enum.output_size
  is_regression = dataset_enum.is_regression

  model = get_model(model_type, variant_settings, input_size, output_size, depths, dropout, num_hidden)

  history = train(model, train_loader, val_loader, is_regression, num_epochs=num_epochs, lr=learning_rate, weight_decay=weight_decay, logger=logger)
  plot_loss_curves(history)

  total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print("Total trainable parameters:", total_params)

  result_metric = eval_model(model, test_loader, is_regression, y_scaler)
  print(f"R2 Score: {result_metric}" if is_regression else f"Accuracy: {result_metric}")

  return result_metric, total_params, model

def train_test_wandb():
    with wandb.init() as run:
        config = run.config
        model_type = Model_Type[config.model_type]
        dataset = Dataset_Enum[config.dataset]

        if config.config_type == "Uniform":
            depths = [config.base_depth] * config.num_ladders
        elif config.config_type == "Even_Odd":
            depths = [config.base_depth if i % 2 == 0 else config.base_depth + 1 for i in range(config.num_ladders)]
        elif config.config_type == "Pyramid":
            depths = [config.base_depth + (i * 2) for i in range(config.num_ladders)]

        logger = Wandb_Logger()
        result_metric, total_params, model = train_test_loop(dataset, model_type, config.variant_settings, depths, config.lr, config.dropout, config.batch_size, num_epochs=50, weight_decay=1e-4, num_hidden=config.num_hidden, logger=logger)
        wandb.log({"score": result_metric, "total_params": total_params})

class Wandb_Logger():
    def __init__(self):
        pass
    def log(self, name, metrics):
        wandb.log({name: metrics})

def train_test_wandb_lim_params():
    with wandb.init() as run:
        config = run.config

        # FIX 1: Variant no longer exists in colognet.py -- use Model_Type instead.
        model_type = Model_Type[config.model_type]

        dataset = Dataset_Enum[config.dataset]
        input_size = dataset.input_size
        output_size = dataset.output_size
        max_params = config.max_params
        num_ladders = config.num_ladders

        if config.config_type == "Uniform":
            remaining = max_params - (output_size * num_ladders + output_size)
            base_depth = floor(remaining / (num_ladders * (input_size + 1)))
            depths = [max(1, base_depth)] * num_ladders
        elif config.config_type == "Even_Odd":
            remaining_budget = max_params - (output_size * num_ladders + output_size)
            base_depth = floor((remaining_budget / (input_size + 1) - (num_ladders // 2)) / num_ladders)
            depths = [max(1, base_depth if i % 2 == 0 else base_depth + 1) for i in range(num_ladders)]
        elif config.config_type == "Pyramid":
            remaining_budget = max_params - (output_size * num_ladders + output_size)
            sum_increments = num_ladders * (num_ladders - 1)
            base_depth = floor((remaining_budget / (input_size + 1) - sum_increments) / num_ladders)
            depths = [max(1, base_depth + (i * 2)) for i in range(num_ladders)]

        # FIX 2: train_test_loop needs variant_settings as its 3rd arg now (old call was
        # missing it, which would've silently shifted every later argument by one slot).
        # FIX 2b (bonus): train_test_loop returns 3 values, this only unpacked 2 --
        # pre-existing bug, unrelated to the variant_settings migration, fixed here too.
        result_metric, total_params, model = train_test_loop(dataset, model_type, config.variant_settings, depths, config.lr, config.dropout, config.batch_size, num_epochs=50, weight_decay=1e-4, num_hidden=config.num_hidden)

        artifact = wandb.Artifact("model-weights", type="model")
        artifact.add_file("model.pt")
        wandb.log_artifact(artifact)
        wandb.log({"score": result_metric, "total_params": total_params})

def get_model(model_type, variant_settings, input_size, output_size, depths, dropout, num_hidden):
    if model_type is Model_Type.MLP:
        target_params = calculate_parameters(input_size, output_size, depths, variant_settings["is_mlp"])
        model = MLP(input_size, output_size, target_params, dropout, num_hidden)
    elif model_type is Model_Type.SWIGLU:
        # FIX 3: calculate_parameters now requires is_mlp with no default.
        # Using variant_settings["is_mlp"] so this baseline stays parameter-matched
        # to whichever ContNet variant it's being compared against.
        target_params = calculate_parameters(input_size, output_size, depths, variant_settings["is_mlp"])
        model = SwiGLUMLP(input_size, output_size, depths, dropout)
    else:
        model = ContNet_Model(variant_settings, input_size, output_size, depths, dropout)
    return model

def eval_model(model, test_loader, is_regression, y_scaler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    num_correct = 0
    num_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if is_regression:
                all_preds.append(outputs.cpu())
                all_labels.append(labels.cpu())
            else:
                _, predicted = torch.max(outputs, 1)
                num_correct += (predicted == labels).sum().item()
                num_total += labels.size(0)

    if is_regression:
        preds = torch.cat(all_preds).numpy()
        targets = torch.cat(all_labels).numpy()
        if y_scaler is not None:
            preds = y_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
            targets = y_scaler.inverse_transform(targets.reshape(-1, 1)).flatten()
        return r2_score(targets, preds)
    else:
        return num_correct / num_total

@staticmethod
def calculate_parameters(input_size, output_size, depths, is_mlp):
  ladder_params_total = sum(d * input_size + d for d in depths)
  combiner_params = output_size * len(depths) + output_size
  target_params = ladder_params_total + combiner_params

  if is_mlp:
      hidden_size = 128
      extra_params_per_ladder = (input_size + sum(depths) + 1) * hidden_size + sum(depths)
      extra_params = extra_params_per_ladder * len(depths)
      target_params = target_params + extra_params

  return target_params