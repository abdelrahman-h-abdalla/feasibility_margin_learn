from common.paths import ProjectPaths
from common.training import train
from common.networks import MultiLayerPerceptron
from common.datasets import TrainingDataset
from common.statistics import binary_predictions, compute_metrics

import os
import socket
import webbrowser
import json

import torch
import torch.nn as nn
import torch.optim as optim
torch.cuda.empty_cache()

from tensorboard import program
from torch.utils.tensorboard import SummaryWriter
import wandb

# Maximum epochs before stopping
EPOCHS = 2000
# Parameters used when not sweeping
BATCH_SIZE_DEFAULT = 2048
LEARNING_RATE_DEFAULT = 0.0001
HIDDEN_LAYERS_DEFAULT = [512, 256, 128]
ACTIVATION_FUNCTION_DEFAULT = 'relu'
SAVE_TRACED_MODEL = False
EVALUATE_STEPS = 50
PATIENCE = 10
LAUNCH_TENSORBOARD = False

class DotDict(dict):
    """ Dictionary that supports dot notation for key access for use with WandB. """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

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
    
def save_network_hyperparameters(hidden_layers, activation_function, run_name, save_directory):
    # Prepare the data t be saved
    network_params = {
        "hidden_layers": hidden_layers,
        "activation_function": activation_function,
    }

    # Create the filename and path
    filename = 'network_hyperparameters_{}.json'.format(run_name)
    file_path = os.path.join(save_directory, filename)

    # Save the data as a JSON file
    with open(file_path, 'w') as f:
        json.dump(network_params, f, indent=4)

    print("Network parameters saved to {}".format(file_path))

def main_train(config=None, stance_legs_str=None):
    robot_name = 'hyqreal'
    paths = ProjectPaths()
    stance_legs = [int(digit) for digit in stance_legs_str if digit.isdigit()]
    data_folder_name = 'stability_margin/' + stance_legs_str + '/'
    network_input_dim = compute_network_input_dim(stance_legs)
    pretrained_model_path = paths.TRAINED_MODELS_PATH + "/final/stability_margin/network_state_dict.pt"

    # WnB configuration
    run_name = 'None'
    if isinstance(config, dict):
        # Initialize WandB with the given configuration
        run = wandb.init(config=config)
        config = wandb.config  # Convert to WandB config
    run_name = run.name

    model_save_dir = paths.TRAINED_MODELS_PATH + '/' + data_folder + '/' + paths.INIT_DATETIME_STR
    try:
        os.makedirs(model_save_dir)
    except OSError:
        pass

    # Save network hyperparameters
    save_network_hyperparameters(config.hidden_layers, config.activation_function, run_name, model_save_dir)
    # Process dataset
    training_dataset_handler = TrainingDataset(data_folder=data_folder_name, robot_name=robot_name,
                                                in_dim=network_input_dim, no_of_stance=stance_legs.count(1))
    data_parser = training_dataset_handler.get_training_data_parser(max_files=800)
    data_folder = training_dataset_handler.get_data_folder()

    # Use torch data_loader to sample training data
    dataloader_params = {'batch_size': config.batch_size, 'shuffle': True, 'num_workers': 12}
    training_dataloader = data_parser.torch_dataloader(dataloader_params, data_type='training')
    validation_dataloader = data_parser.torch_dataloader(dataloader_params, data_type='validation')

    # Initialize Network Object
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network = MultiLayerPerceptron(in_dim=network_input_dim, out_dim=1, hidden_layers=config.hidden_layers, activation=config.activation_function)
    # network = MultiLayerPerceptron(in_dim=network_input_dim, out_dim=1, hidden_layers=config.hidden_layers)
    network.to(device)

    # Load pretrained model weights
    # network.load_state_dict(torch.load(pretrained_model_path, map_location=device))

    optimizer = optim.Adam(network.parameters(), lr=config.learning_rate)
    print('\nTraining for', network.get_trainable_parameters(), 'network parameters.\n')

    # TensorBoard
    log_path = paths.LOGS_PATH + '/' + data_folder + '/' + paths.INIT_DATETIME_STR
    writer = SummaryWriter(log_path)

    if LAUNCH_TENSORBOARD:
        tensorboard_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tensorboard_socket.bind(('', 0))
        tensorboard_port = tensorboard_socket.getsockname()[1]
        tensorboard_socket.close()

        tensorboard_launcher = program.TensorBoard()
        tensorboard_launcher.configure(
            argv=[None, '--logdir', log_path, '--port', str(tensorboard_port)])
        tensorboard_address = tensorboard_launcher.launch()
        webbrowser.open_new_tab(tensorboard_address + '#scalars&_smoothingWeight=0')

    # Network Save Path
    save_path = model_save_dir + '/network_state_dict_' + run_name + '.pt'

    # Train and Validate
    iterator_offset = train(training_dataloader, validation_dataloader, device, optimizer, network, writer,
                            config.batch_size, EPOCHS, EVALUATE_STEPS, PATIENCE, save_path=save_path)

    if SAVE_TRACED_MODEL:
        save_path = model_save_dir + '/traced_network_model.pt'

        params = {'batch_size': 1, 'shuffle': True, 'num_workers': 1}
        tracing_dataloader = data_parser.torch_dataloader(params=params, data_type='testing')

        for local_batch, local_targets in tracing_dataloader:
            traced_network = torch.jit.trace(network.cpu(), local_batch.cpu())
            break

        traced_network.cpu().save(save_path)

    # Testing
    network.to(device).eval()
    dataloader_params = {'batch_size': int(config.batch_size), 'shuffle': True, 'num_workers': 12}
    testing_dataloader = data_parser.torch_dataloader(params=dataloader_params, data_type='testing')

    test_loss = 0.0
    sub_epoch_iterator = 0
    prediction_iterator = 0
    iterator = iterator_offset - 1
    # Initialize overall TP, TN, FP, FN counters
    overall_TP = overall_TN = overall_FP = overall_FN = 0

    for local_batch, local_targets in testing_dataloader:
        local_batch, local_targets = local_batch.to(device), local_targets.to(device)

        output = network(local_batch)

        loss = nn.MSELoss()(output, local_targets)
        test_loss += (loss.item())

        # Convert to binary predictions
        preds = binary_predictions(output)
        # Compute metrics for the current batch
        TP, TN, FP, FN, accuracy = compute_metrics(preds, local_targets)

        # Update overall counters
        overall_TP += TP
        overall_TN += TN
        overall_FP += FP
        overall_FN += FN

        if sub_epoch_iterator % EVALUATE_STEPS == EVALUATE_STEPS - 1:
            print('[Test, %d, %5d] loss: %.8f' % (1, sub_epoch_iterator + 1, test_loss / EVALUATE_STEPS))
            iterator += 1

            writer.add_scalars('Loss', {'Test': test_loss / EVALUATE_STEPS}, iterator)
            # Log to WandB
            if wandb.run is not None:
                wandb.log({'Test Loss': test_loss / EVALUATE_STEPS}, step=iterator)
            test_loss = 0.0

            if prediction_iterator < EVALUATE_STEPS:
                writer.add_scalars('Prediction', {'Target': local_targets[-1].item() / 10.0},
                                   prediction_iterator)
                writer.add_scalars('Prediction', {'NetworkOutput': output[-1].item() / 10.0},
                                   prediction_iterator)
                if wandb.run is not None:
                    wandb.log({'Target (last of batch)': local_targets[-1].item() / 10.0, 'NetworkOutput (last of batch)': output[-1].item() / 10.0}, step=prediction_iterator)
                prediction_iterator += 1
            writer.flush()

        sub_epoch_iterator += 1

    # Compute overall accuracy
    total_samples = overall_TP + overall_TN + overall_FP + overall_FN
    overall_accuracy = (overall_TP + overall_TN) / total_samples if total_samples > 0 else 0

    # Print overall TP, TN, FP, FN, and Accuracy
    print("True Positives (TP): {}".format(overall_TP))
    print("True Negatives (TN): {}".format(overall_TN))
    print("False Positives (FP): {}".format(overall_FP))
    print("False Negatives (FN): {}".format(overall_FN))
    print("Accuracy: {:.8f}".format(overall_accuracy))

    # Optionally log overall metrics as well
    if wandb.run is not None:
        wandb.log({
            'True Positives': overall_TP,
            'True Negatives': overall_TN,
            'False Positives': overall_FP,
            'False Negatives': overall_FN,
            'Accuracy': overall_accuracy
        })

    print("Tested!")
    
    # while True:
    #     try:
    #         pass
    #     except KeyboardInterrupt:
    #         exit(0)

def run_sweep(stance_legs_str):
    sweep_config = {
        'method': 'random',  # Could be 'grid' or 'bayesian'
        'metric': {
            'name': 'Validation Loss',
            'goal': 'minimize'
        },
        'parameters': {
            'learning_rate': {
                'min': 0.0001,
                'max': 0.001
            },
            'batch_size': {
                'values': [1024, 2048, 4096]
            },
            'hidden_layers': {
                'values': [[256, 128, 128], [512, 256, 128], [256, 256, 128]]
            },
            'activation_function': {
                'values': ['relu', 'softsign']
            }
        }
    }
    sweep_id = wandb.sweep(sweep_config, project='FM')
    wandb.agent(sweep_id, lambda: main_train(sweep_config, stance_legs_str))

def run_single(stance_legs_str):
    config = DotDict({
        'learning_rate': LEARNING_RATE_DEFAULT,
        'batch_size': BATCH_SIZE_DEFAULT,
        'hidden_layers': HIDDEN_LAYERS_DEFAULT,
        'activation_function': ACTIVATION_FUNCTION_DEFAULT
    })
    main_train(config, stance_legs_str)

def main(sweep=False):
    stance_legs_str = input("Enter Stance Legs [xxxx]: ")
    if sweep:
        run_sweep(stance_legs_str)
    else:
        run_single(stance_legs_str)

if __name__ == '__main__':
    # Set use_sweep to True if you want to perform a hyperparameter sweep
    main(sweep=True)