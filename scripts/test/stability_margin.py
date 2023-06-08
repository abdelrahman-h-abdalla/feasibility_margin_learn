import numpy as np

import torch
import torch.nn.functional as F

from common.paths import ProjectPaths
from common.datasets import TrainingDataset
from common.networks import MultiLayerPerceptron


def main():
    paths = ProjectPaths()
    dataset_handler = TrainingDataset()

    model_directory = '/final/stability_margin/'
    model_directory = paths.TRAINED_MODELS_PATH + model_directory
    print("model directory: ", model_directory)

    network = MultiLayerPerceptron(in_dim=47, out_dim=1, hidden_layers=[256,128,128], activation=F.softsign, dropout=0.0)
    network.load_state_dict(torch.load(model_directory + 'network_state_dict.pt'))
    # network.load_params_from_txt(model_directory + 'network_parameters_anymal_c.txt')


    network.eval()

    delta_pos_range = 0.6
    num_of_tests = 25
    delta_pos_range_vec = np.linspace(-delta_pos_range/2.0, delta_pos_range/2.0, num_of_tests)
    print('Stability Margin X:')
    for delta in delta_pos_range_vec:
        network_input = np.concatenate([
            np.array([0.0, 0.0, 1.0]),
            np.zeros(15),
            np.array([0.3 - delta, 0.2, -0.4]),
            np.array([0.3 - delta, -0.2, -0.4]),
            np.array([-0.3 - delta, 0.2, -0.4]),
            np.array([-0.3 - delta, -0.2, -0.4]),
            np.array([0.5]),
            np.ones(4),
            np.array([0.0, 0.0, 1.0] * 4)
        ])
        network_input = dataset_handler.scale_input(network_input)
        output = network(torch.from_numpy(network_input).float()).item()
        output = dataset_handler.scale_output(output)
        print(output)

    print('\n------------------------')
    print('Stability Margin X LH:')
    for delta in delta_pos_range_vec:
        network_input = np.concatenate([
            np.array([0.0, 0.0, 1.0]),
            np.zeros(15),
            np.array([0.3 - delta, 0.2, -0.4]),
            np.array([0.3 - delta, -0.2, -0.4]),
            np.array([-0.3 - delta, 0.2, -0.4]),
            np.array([-0.3 - delta, -0.2, -0.4]),
            np.array([0.5]),
            np.ones(2),
            np.zeros(1),
            np.ones(1),
            np.array([0.0, 0.0, 1.0] * 4)
        ])
        network_input = dataset_handler.scale_input(network_input)
        output = network(torch.from_numpy(network_input).float()).item()
        output = dataset_handler.scale_output(output)
        print(output)

    print('\n------------------------')
    print('Stability Margin Y LH:')
    for delta in delta_pos_range_vec:
        network_input = np.concatenate([
            np.array([0.0, 0.0, 1.0]),
            np.zeros(15),
            np.array([0.3, 0.2 - delta, -0.4]),
            np.array([0.3, -0.2 - delta, -0.4]),
            np.array([-0.3, 0.2 - delta, -0.4]),
            np.array([-0.3, -0.2 - delta, -0.4]),
            np.array([0.5]),
            np.ones(2),
            np.zeros(1),
            np.ones(1),
            np.array([0.0, 0.0, 1.0] * 4)
        ])
        network_input = dataset_handler.scale_input(network_input)
        output = network(torch.from_numpy(network_input).float()).item()
        output = dataset_handler.scale_output(output)
        print(output)

    print('\n------------------------')
    print('Stability Margin Y:')
    for delta in delta_pos_range_vec:
        network_input = np.concatenate([
            np.array([0.0, 0.0, 1.0]),
            np.zeros(15),
            np.array([0.3 , 0.2 - delta, -0.4]),
            np.array([0.3 , -0.2 - delta, -0.4]),
            np.array([-0.3, 0.2 - delta, -0.4]),
            np.array([-0.3, -0.2 - delta, -0.4]),
            np.array([0.5]),
            np.ones(4),
            np.array([0.0, 0.0, 1.0] * 4)
        ])
        network_input = dataset_handler.scale_input(network_input)
        output = network(torch.from_numpy(network_input).float()).item()
        output = dataset_handler.scale_output(output)
        print(output)

    delta_acc_range = 8.0
    delta_acc_range_vec = np.linspace(-delta_acc_range/2.0, delta_acc_range/2.0, num_of_tests)

    print('\n------------------------')
    print('Stability Margin X ACC:')
    for delta in delta_acc_range_vec:
        network_input = np.concatenate([
            np.array([0.0, 0.0, 1.0]),
            np.zeros(3),
            np.array([0. + delta, 0., 0.]),
            np.zeros(9),
            np.array([0.3, 0.2, -0.4]),
            np.array([0.3, -0.2, -0.4]),
            np.array([-0.3, 0.2, -0.4]),
            np.array([-0.3, -0.2, -0.4]),
            np.array([0.5]),
            np.zeros(1),
            np.ones(2),
            np.zeros(1),
            np.array([0.0, 0.0, 1.0] * 4)
        ])
        network_input = dataset_handler.scale_input(network_input)
        output = network(torch.from_numpy(network_input).float()).item()
        output = dataset_handler.scale_output(output)
        print(output)

    print('\n------------------------')
    print('Stability Margin Y ACC:')
    for delta in delta_acc_range_vec:
        network_input = np.concatenate([
            np.array([0.0, 0.0, 1.0]),
            np.zeros(3),
            np.array([0., 0. + delta, 0.]),
            np.zeros(9),
            np.array([0.3, 0.2, -0.4]),
            np.array([0.3, -0.2, -0.4]),
            np.array([-0.3, 0.2, -0.4]),
            np.array([-0.3, -0.2, -0.4]),
            np.array([0.5]),
            np.zeros(1),
            np.ones(2),
            np.zeros(1),
            np.array([0.0, 0.0, 1.0] * 4)
        ])
        network_input = dataset_handler.scale_input(network_input)
        output = network(torch.from_numpy(network_input).float()).item()
        output = dataset_handler.scale_output(output)
        print(output)


if __name__ == '__main__':
    main()

