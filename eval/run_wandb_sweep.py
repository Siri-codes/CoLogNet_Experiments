"""
Weights & Biases Testing Script

"""
import wandb
import os
from CoLogNet_Experiments.eval.colognet_test_loops import train_test_wandb

COLOGNET = {
    'is_cofr' : False,
    'is_bin' : False,
    'is_orig' : False,
    'is_seq' : False,
    'is_euc_dist' : False,
    'is_reciprocal' : False,
    'is_dist_norm' : False
}

COLOGNET_E = {
    'is_cofr' : False,
    'is_bin' : False,
    'is_orig' : False,
    'is_seq' : False,
    'is_euc_dist' : True,
    'is_reciprocal' : False,
    'is_dist_norm' : False
}

COLOGNET_S = {
    'is_cofr' : False,
    'is_bin' : False,
    'is_orig' : False,
    'is_seq' : True,
    'is_euc_dist' : False,
    'is_reciprocal' : False,
    'is_dist_norm' : False
}

#some useful sweep configurations:
sweep_config_bayes_mlp = {
    'method': 'bayes',  # Grid search: iterates through all combinations
    'metric': {'name': 'score', 'goal': 'maximize'},  # maximizing accuracy
    'parameters': {
        'dataset': {'values': ['MNIST']},
        'model_type': {'values': ['MLP', 'CONTNET']},
        'variant_settings': {'values': [COLOGNET, COLOGNET_E, COLOGNET_S]}
        'lr': {'distribution': 'log_uniform_values',
               'min': 0.001,
               'max': 0.05},
        'num_ladders' : {'values': [1]}, #for mlp num_ladders doesn't matter
        'max_params': {'values': [6000]},  # restrict # of total parameters model can use
        'config_type': {'values': ['Even_Odd']},
        'batch_size': {'values': [64, 128, 256, 512]},
        'dropout': {'values': [0.0, 0.1, 0.2]},
        'num_hidden': {'values': [1, 2, 3]}
    }
}
sweep_config_grid = {
    'method': 'grid',  # Grid search: iterates through all combinations
    'metric': {'name': 'score', 'goal': 'maximize'},  # maximizing accuracy
    'parameters': {
        'dataset': {'values': ['MNIST']},
        'model_type': {'values': ['MLP', 'CONTNET']},
        'lr': {'values': [0.001, 0.005, 0.01, 0.05]},
        'num_ladders' : {'values': [3, 4, 7, 8, 11, 12, 15, 16]}, #for mlp num_ladders doesn't matter
        'base_depth': {'values': [3, 4, 7, 8]},  # restrict # of total parameters model can use
        'config_type': {'values': ['Even_Odd']},
        'batch_size': {'values': [128, 256, 512]},
        'dropout': {'values': [0.0, 0.1, 0.2]},
        'num_hidden': {'values': [1]}
    }
}

WANDB_ENTITY = "colognet-project" 
WANDB_PROJECT = "CoLogNet_Experiments"
sweep_file = "sweep_id.txt"

def get_or_create_sweep_id():
    '''
    Retrieves an existing W&B sweep ID stored in sweep_file or creates a new sweep if file doesn't exist.
    '''

    # Check if sweep_id already exists (in sweep_file)
    if os.path.exists(sweep_file):
        with open(sweep_file, "r") as f:
            sweep_id = f.read().strip()

        # Format the ID to include entity and project to ensure it is found
        full_sweep_id = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{sweep_id}"
        print(f">>> Resuming existing sweep: {full_sweep_id}")
        return full_sweep_id

    # Choose which parameters to iterate over:
    sweep_config = sweep_config_grid

    # If not found, create a new sweep
    sweep_id = wandb.sweep(sweep_config, project="CoLogNet_Experiments")
    
    with open(sweep_file, 'w') as f: # save sweep_id for future runs
        f.write(sweep_id)

    print(f"Created new sweep ID: {sweep_id}")

    # For the first run, return the full path
    return f"{WANDB_ENTITY}/{WANDB_PROJECT}/{sweep_id}"

def main():
    '''
    Run the W&B sweep
    '''
    sweep_id = get_or_create_sweep_id()
    wandb.agent(sweep_id, function=train_test_wandb) #runs the hyperparameter sweep

if __name__ == "__main__":
    main()