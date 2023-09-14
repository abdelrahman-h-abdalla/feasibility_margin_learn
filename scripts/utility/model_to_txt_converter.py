import os
import sys
from datetime import datetime
import shutil
import numpy as np

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

def main():
    robot_name = 'hyqreal'
    paths = ProjectPaths()

    model_directory = get_latest_directory(paths.TRAINED_MODELS_PATH + '/stability_margin/')
    model_directory = paths.TRAINED_MODELS_PATH + "/stability_margin/" + model_directory + "/"
    print("Loading model from", model_directory)
    save_directory = '/final/stability_margin/'
    save_directory = paths.TRAINED_MODELS_PATH + save_directory
    # Create the destination directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    network = MultiLayerPerceptron(in_dim=47, out_dim=1, hidden_layers=[256, 128, 128], activation=F.softsign, dropout=0.0)
    network.load_state_dict(torch.load(model_directory + 'network_state_dict.pt'))

    model_parameters = list(network.state_dict().keys())
    model_parameters = np.concatenate(
        [network.state_dict()[key].cpu().numpy().reshape(-1) for key in model_parameters])

    print("Saving model in:", save_directory)
    param_save_name = save_directory + 'network_parameters_' + robot_name + '.txt'
    np.savetxt(param_save_name, model_parameters.reshape((1, -1)), delimiter=', ',
               newline='\n', fmt='%1.10f')
    copy_file(model_directory + 'network_state_dict.pt', save_directory)

    print('\nSaved model parameters in the following order:')
    for parameter_key in list(network.state_dict().keys()):
        print('   ', parameter_key, '| Dimension:', network.state_dict()[parameter_key].shape)


if __name__ == '__main__':
    main()
