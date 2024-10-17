import os
import sys
from datetime import datetime
import shutil
import numpy as np
import json

import torch
import torch.nn.functional as F

from common.paths import ProjectPaths
from common.networks import MultiLayerPerceptron


def get_latest_directory(root_path):
    # Get list of directories in the current working directory
    directories = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    # Filter directories that don't match the time format
    time_format = "%Y-%m-%d-%H-%M-%S"
    time_directories = []
    for d in directories:
        try:
            datetime.strptime(d, time_format)
            time_directories.append(d)
        except ValueError:
            continue

    # Sort directories by date and return the latest
    time_directories.sort(key=lambda date: datetime.strptime(date, time_format), reverse=True)

    return time_directories[0] if time_directories else None

def copy_file(src_path, dst_path):
    try:
        shutil.copy(src_path, dst_path)
    except IOError as e:
        print("Unable to copy file.", e)
    except:
        print("Unexpected error:", sys.exc_info())

def compute_network_input_dim(stance_legs):
    no_of_stance = stance_legs.count(1)
    if no_of_stance == 4:
        return 40
    elif no_of_stance == 3:
        return 34
    elif no_of_stance == 2:
        return 28
    else:
        print("Wrong number of inputs")
        return None
    
def load_hyperparameters_from_json(model_directory):
    # Find the .json file in the directory
    json_files = [f for f in os.listdir(model_directory) if f.endswith('.json')]
    if not json_files:
        raise FileNotFoundError("No .json hyperparameters file found in the directory.")

    # Assuming there's only one .json file, load it
    json_path = os.path.join(model_directory, json_files[0])
    try:
        with open(json_path, 'r') as f:
            hyperparemters = json.load(f)
        print("Loaded hyperparameters from:", json_path)
        return hyperparemters
    except json.JSONDecodeError:
        raise ValueError("Error decoding JSON file at {}".format(json_path))

def main():
    robot_name = 'hyqreal'
    stance_legs_str = input("Enter Stance Legs [xxxx]: ")
    stance_legs = [int(digit) for digit in stance_legs_str if digit.isdigit()]
    paths = ProjectPaths()

    model_directory = get_latest_directory(paths.TRAINED_MODELS_PATH + '/stability_margin/' + stance_legs_str + '/')
    model_directory = paths.TRAINED_MODELS_PATH + "/stability_margin/" + stance_legs_str + "/" + model_directory + "/"    
    print("Loading model", model_directory + os.listdir(model_directory)[0])

    # Load hyperparameters from the JSON file
    hyperparameters = load_hyperparameters_from_json(model_directory)
    # Ensure that hyperparameters are present in the JSON file
    hidden_layers = hyperparameters['hidden_layers']
    activation_function = hyperparameters['activation_function']

    save_directory = '/final/stability_margin/'
    save_directory = paths.TRAINED_MODELS_PATH + save_directory
    # Create the destination directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    network_input_dim = compute_network_input_dim(stance_legs)
    network = MultiLayerPerceptron(in_dim=network_input_dim, out_dim=1, hidden_layers=hidden_layers,
                                   activation=activation_function, dropout=0.0)
    pt_files = [f for f in os.listdir(model_directory) if f.endswith('.pt')]
    if not pt_files:
        raise FileNotFoundError("No .pt file found in the directory")
    
    # Load the first .pt file found
    model_path = os.path.join(model_directory, pt_files[0])
    # model_path = model_directory + os.listdir(model_directory)[0]
    network.load_state_dict(torch.load(model_path))

    model_parameters = list(network.state_dict().keys())
    model_parameters = np.concatenate(
        [network.state_dict()[key].cpu().numpy().reshape(-1) for key in model_parameters])

    print("Saving model in:", save_directory)
    param_save_name = save_directory + 'network_parameters_' + robot_name + '.txt'
    np.savetxt(param_save_name, model_parameters.reshape((1, -1)), delimiter=', ',
               newline='\n', fmt='%1.10f')
    copy_file(model_path, save_directory)

    print('\nSaved model parameters in the following order:')
    for parameter_key in list(network.state_dict().keys()):
        print('   ', parameter_key, '| Dimension:', network.state_dict()[parameter_key].shape)


if __name__ == '__main__':
    main()
