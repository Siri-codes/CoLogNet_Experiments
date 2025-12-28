#W&B TESTING SCRIPT:

import wandb
import os
from eval.colognet_test_loops import train_test_wandb

sweep_file = "sweep_id.txt"

def get_or_create_sweep_id():
    '''
    Retrieves an existing W&B sweep ID stored in sweep_file or creates a new sweep if file doesn't exist.
    '''

    # Check if sweep_id already exists (in sweep_file)
    if os.path.exists(sweep_file):
        with open(sweep_file, "r") as f:
            sweep_id = f.read().strip()
        print(f">>> Resuming existing sweep: {sweep_id}")
        return sweep_id

    # Choose which parameters to iterate over:
    sweep_config = {
        'method': 'grid',  # Grid search: iterates through all combinations
        'metric': {'name': 'accuracy', 'goal': 'maximize'}, # maximizing accuracy
        'parameters': {
            'dataset': {'values': ['MNIST', 'WAVEFORM', 'BOSTON']},
            'model_type': {'values': ['MLP', 'SWIGLU', 'COFRNET','COLOGNET', 'COLOGNET_B']},
            'lr': {'values': [0.0005, 0.001, 0.005, 0.01]},
            'num_ladders': {'values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
            'base_depth': {'values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}, 
            'config_type': {'values': ['Uniform', 'Even_Odd', 'Pyramid']},
            'batch_size': {'values': [32, 64, 128]},
            'dropout': {'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]},
        }
    }

    # If not found, create a new sweep
    sweep_id = wandb.sweep(sweep_config, project="CoLogNet Experiments")
    
    with open(sweep_file, 'w') as f: # save sweep_id for future runs
        f.write(sweep_id)

    print(f"Created new sweep ID: {sweep_id}")
    return sweep_id


# Initialize and start
sweep_id = get_or_create_sweep_id()
wandb.agent(sweep_id, function=train_test_wandb) #runs the hyperparameter sweep