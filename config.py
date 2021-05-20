import torch
torch.set_num_threads(1)

CONFIG_FASHION_MNIST = {
    "nb_epochs": 10,
    "nb_inputs": 28 * 28,
    "nb_hidden": 200,
    "nb_hidden2": 200,
    "nb_outputs": 10,
    "nb_steps": 100,
    "batch_size": 256,
    "time_step": 1e-3,
    "tau_mem": 10e-3,
    "tau_readout": 10e-3,
    "tau_eff": 20e-3,
    "th": 1.,  # Regular threshold
    "b_th": 0.8,  # Backprop threshold, in the paper B_th = 1.-b_th
    "lr": 2e-4,
    "betas": (0.9, 0.999),
    "seed": 0,
    "set_random_state": False,
    "weight_multiplier": 20.,
    "dtype": torch.float,
    "device": "cuda",
    "dataset_path": "~/data/datasets/torch/fashion-mnist",
    "dataset_id": 'fmnist',
    "dataset_name": 'Fashion-MNIST',
    "dataset_type": 'torchvision',
    "PREFIX": 'ERROR',
    "lambda_upper": 0.06,
    "lambda_lower": 100.,
    "f_upper": 1.,  # Average number of spikes per hidden neuron in nb_steps
    "f_lower": 1e-3,
    "p_up": 1,
}


CONFIG_N_MNIST = {
    "nb_epochs": 10,
    "nb_inputs": 34 * 34,
    "nb_hidden": 200,
    "nb_hidden2": 200,  # 50
    "nb_outputs": 10,
    "nb_steps": 300,  # 100
    "batch_size": 256,  # 256
    "time_step": 1e-3,
    "tau_mem": 10e-3,
    "tau_readout": 10e-3,
    "tau_eff": None,
    "th": 1.,  # Regular threshold
    "b_th": 0.8,  # Backprop threshold, in the paper B_th = 1.-b_th
    "lr": 2e-4,
    "betas": (0.9, 0.999),
    "seed": 0,
    "set_random_state": False,
    "weight_multiplier": None,
    "dtype": torch.float,
    "device": "cuda",
    "dataset_path": "datasets/nmnist",
    "dataset_id": 'nmnist',
    "dataset_name": 'Neuromorphic-MNIST',
    "dataset_type": 'h5',
    "PREFIX": 'ERROR',
    "lambda_upper": 0.06,
    "lambda_lower": 100.,
    "f_upper": 1.,  # Average number of spikes per hidden neuron in nb_steps
    "f_lower": 1e-3,
    "p_up": 1,
}


CONFIG_SHD = {
    "nb_epochs": 10,
    "nb_inputs": 700,
    "nb_hidden": 200,
    "nb_hidden2": 200,
    "nb_outputs": 20,
    "nb_steps": 500,
    "batch_size": 256,
    "time_step": 2e-3,
    "tau_mem": 10e-3,
    "tau_readout": 20e-3,
    "tau_eff": None,
    "th": 1.,  # Regular threshold
    "b_th": 0.8,  # Backprop threshold, in the paper B_th = 1.-b_th
    "lr": 1e-3,
    "betas": (0.9, 0.999),
    "seed": 0,
    "set_random_state": False,
    "weight_multiplier": None,
    "dtype": torch.float,
    "device": "cuda",
    "dataset_path": "/SHD",
    "dataset_type": 'h5',
    "dataset_id": 'SHD',
    "dataset_name": 'datasets/SHD',
    "PREFIX": 'ERROR',
    "lambda_upper": 0.06,
    "lambda_lower": 100.,
    "f_upper": 10.,  # Average number of spikes per hidden neuron in nb_steps
    "f_lower": 1e-3,
    "p_up": 1,
}

CONFIGS = {'fmnist': CONFIG_FASHION_MNIST, 'nmnist': CONFIG_N_MNIST, 'SHD': CONFIG_SHD}





