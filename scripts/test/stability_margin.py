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
    # network.load_state_dict(torch.load(model_directory + 'network_state_dict.pt'))
    network.load_params_from_txt(model_directory + 'network_parameters_anymal_c.txt')


    network.eval()

    delta_pos_range = 0.6
    num_of_tests = 25
    delta_pos_range_vec = np.linspace(-delta_pos_range/2.0, delta_pos_range/2.0, num_of_tests)
    out_array = []
    jac_array = []
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
        network_input = torch.tensor(network_input, requires_grad=True) # requires_grad to record gradients w.r.t input
        output = network(network_input.float())
        # output = network(torch.from_numpy(network_input).float()).item()
        output = dataset_handler.scale_output(output)
        out_array.append(output.detach().numpy()[:][0])
        # Compute gradients of the output w.r.t. the input
        output.backward(torch.ones_like(output))
        unnormalized_gradient = dataset_handler.unscale_gradient(network_input.grad.numpy())
        jac_array.append(-1 * (unnormalized_gradient[18] + unnormalized_gradient[21] + \
            unnormalized_gradient[24] + unnormalized_gradient[27]))
    print('Stability Margin X:')
    for a in out_array:
        print(a)
    print("Jacobian:")
    for j in jac_array:
        print(j)

    print('\n------------------------')
    print('Stability Margin Y:')
    out_array = []
    jac_array = []
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
        network_input = torch.tensor(network_input, requires_grad=True) # requires_grad to record gradients w.r.t input
        output = network(network_input.float())
        # output = network(torch.from_numpy(network_input).float()).item()
        output = dataset_handler.scale_output(output)
        out_array.append(output.detach().numpy()[:][0])
        # Compute gradients of the output w.r.t. the input
        output.backward(torch.ones_like(output))
        unnormalized_gradient = dataset_handler.unscale_gradient(network_input.grad.numpy())
        jac_array.append(-1 * (unnormalized_gradient[19] + unnormalized_gradient[22] + \
            unnormalized_gradient[25] + unnormalized_gradient[28]))
    print('Stability Margin Y:')
    for a in out_array:
        print(a)
    print("Jacobian:")
    for j in jac_array:
        print(j)

    delta_acc_range = 8.0
    delta_acc_range_vec = np.linspace(-delta_acc_range/2.0, delta_acc_range/2.0, num_of_tests)

    print('\n------------------------')
    out_array = []
    jac_array = []
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
        network_input = torch.tensor(network_input, requires_grad=True) # requires_grad to record gradients w.r.t input
        output = network(network_input.float())
        # output = network(torch.from_numpy(network_input).float()).item()
        output = dataset_handler.scale_output(output)
        out_array.append(output.detach().numpy()[:][0])
        # Compute gradients of the output w.r.t. the input
        output.backward(torch.ones_like(output))
        unnormalized_gradient = dataset_handler.unscale_gradient(network_input.grad.numpy())
        jac_array.append(-1 * (unnormalized_gradient[18] + unnormalized_gradient[21] + \
            unnormalized_gradient[24] + unnormalized_gradient[27]))
    print('Stability Margin X LH:')
    for a in out_array:
        print(a)
    print("Jacobian:")
    for j in jac_array:
        print(j)

    print('\n------------------------')
    out_array = []
    jac_array = []
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
        network_input = torch.tensor(network_input, requires_grad=True) # requires_grad to record gradients w.r.t input
        output = network(network_input.float())
        # output = network(torch.from_numpy(network_input).float()).item()
        output = dataset_handler.scale_output(output)
        out_array.append(output.detach().numpy()[:][0])
        # Compute gradients of the output w.r.t. the input
        output.backward(torch.ones_like(output))
        unnormalized_gradient = dataset_handler.unscale_gradient(network_input.grad.numpy())
        jac_array.append(-1 * (unnormalized_gradient[19] + unnormalized_gradient[22] + \
            unnormalized_gradient[25] + unnormalized_gradient[28]))
    print('Stability Margin Y LH:')
    for a in out_array:
        print(a)
    print("Jacobian:")
    for j in jac_array:
        print(j)

    print('\n------------------------')
    out_array = []
    jac_array = []
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
        network_input = torch.tensor(network_input, requires_grad=True) # requires_grad to record gradients w.r.t input
        output = network(network_input.float())
        # output = network(torch.from_numpy(network_input).float()).item()
        output = dataset_handler.scale_output(output)
        out_array.append(output.detach().numpy()[:][0])
        # Compute gradients of the output w.r.t. the input
        output.backward(torch.ones_like(output))
        unnormalized_gradient = dataset_handler.unscale_gradient(network_input.grad.numpy())
        jac_array.append(unnormalized_gradient[6])
    print('Stability Margin X ACC:')
    for a in out_array:
        print(a)
    print("Jacobian:")
    for j in jac_array:
        print(j)

    print('\n------------------------')
    jacs = []
    out_array = []
    jac_array = []
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
        network_input = torch.tensor(network_input, requires_grad=True) # requires_grad to record gradients w.r.t input
        output = network(network_input.float())
        # output = network(torch.from_numpy(network_input).float()).item()
        output = dataset_handler.scale_output(output)
        out_array.append(output.detach().numpy()[:][0])
        # Compute gradients of the output w.r.t. the input
        output.backward(torch.ones_like(output))
        unnormalized_gradient = dataset_handler.unscale_gradient(network_input.grad.numpy())
        jac_array.append(unnormalized_gradient[7])
        # jacs.append()
    print('Stability Margin Y ACC:')
    for a in out_array:
        print(a)
    print("Jacobian:")
    for j in jac_array:
        print(j)

    # print('\n------------------------')
    # print('X Jacobian:')
    # network_input = np.concatenate([
    #     np.array([0.0, 0.0, 1.0]),
    #     np.zeros(15),
    #     np.array([0.3, 0.2, -0.4]),
    #     np.array([0.3, -0.2, -0.4]),
    #     np.array([-0.3, 0.2, -0.4]),
    #     np.array([-0.3, -0.2, -0.4]),
    #     np.array([0.5]),
    #     np.ones(4),
    #     np.array([0.0, 0.0, 1.0] * 4)
    #     ])
    # network_input = dataset_handler.scale_input(network_input)
    # network_input = torch.tensor(network_input, requires_grad=True) # requires_grad to record gradients w.r.t input
    # output = network(network_input.float())
    # output = dataset_handler.scale_output(output)
    # # Compute gradients of the output w.r.t. the input
    # output.backward(torch.ones_like(output))
    # # Now the gradients are stored in network_input.grad
    # # print(network_input.grad.numpy())
    # # jac = network_input.grad.numpy()[18:21] + network_input.grad.numpy()[21:24] + \
    # #         network_input.grad.numpy()[24:27] + network_input.grad.numpy()[27:30]
    # jac = -1 * (network_input.grad.numpy()[18] + network_input.grad.numpy()[21] + \
    #         network_input.grad.numpy()[24] + network_input.grad.numpy()[27])
    # print(jac)


if __name__ == '__main__':
    main()
