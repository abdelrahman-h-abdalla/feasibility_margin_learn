import torch
import torch.nn as nn

import itertools


def train(training_dataloader, validation_dataloader, device, optimizer, network, writer, batch_size=64, epochs=8,
          evaluate_steps=50, patience=5, save_path=None):
    iterator = 0

    best_val_loss = float('inf')
    best_epoch_val_loss = float('inf')
    patience_counter = 0
    best_model = None  # To store the best model over mini batches
    best_epoch_model = None # To store the best model over all epochs

    for epoch in range(epochs):
        training_loss = 0.0
        validation_loss = 0.0
        epoch_validation_loss = 0.0

        sub_epoch_iterator = 0

        # Train and Validate iteratively over all mini-batches
        for training_iterable, validation_iterable in zip(training_dataloader,
                                                                       itertools.cycle(validation_dataloader)):
            torch.set_printoptions(precision=10)
            
            training_batch, training_targets = training_iterable[0].to(device), training_iterable[1].to(device)
            network.train()

            optimizer.zero_grad()
            output = network(training_batch)

            loss = nn.MSELoss()(output, training_targets)

            loss.backward()
            training_loss += (loss.item()) # accumulated loss of training mini-batches

            optimizer.step()

            # Validation Loop
            with torch.no_grad():
                validation_batch, validation_targets = validation_iterable[0].to(device), validation_iterable[1].to(
                    device)
                network.eval()

                output = network(validation_batch)
                loss = nn.MSELoss()(output, validation_targets)

                validation_loss += (loss.item()) # accumulated loss validation mini-batches
                epoch_validation_loss += (loss.item()) # accumulated loss of all validation batches in each epoch
                
                if sub_epoch_iterator % evaluate_steps == evaluate_steps - 1:  # update every 50 mini-batches
                    print(
                            '[Training, %d, %5d] loss: %.8f' % (
                        epoch + 1, sub_epoch_iterator + 1, training_loss / evaluate_steps) # average loss over x mini-batches
                    )
                    print(
                            '[Validation, %d, %5d] loss: %.8f' % (
                        epoch + 1, sub_epoch_iterator + 1, validation_loss / evaluate_steps)
                    )

                    writer.add_scalars('Loss', {'Training': training_loss / evaluate_steps}, iterator)
                    writer.add_scalars('Loss', {'Validation': validation_loss / evaluate_steps}, iterator)
                    writer.flush()

                    # Check loss every x mini-batches and update best model if it improved
                    if validation_loss / evaluate_steps < best_val_loss:
                        best_val_loss = validation_loss / evaluate_steps
                        # Save the best model in memory (note it only takes model of only last batch)
                        best_model = network.state_dict()

                    iterator += 1
                    training_loss = 0.0
                    validation_loss = 0.0

                sub_epoch_iterator += 1
        
        print('Avg validation loss of epoch:', epoch_validation_loss / sub_epoch_iterator)
        # Check best loss in last epoch (from all mini-batches)
        if epoch_validation_loss / sub_epoch_iterator < best_epoch_val_loss:
            best_epoch_val_loss = epoch_validation_loss / sub_epoch_iterator
            best_epoch_model = best_model
            patience_counter = 0  # reset counter
        else:
            patience_counter += 1  # increment counter
            
        # If performance didn't improve for 'patience' number of epochs, stop training
        if patience_counter >= patience:
            print("Early stopping due to lack of improvement in validation loss")
            print("Trained!")
            if save_path is not None and best_model is not None:
                torch.save(best_epoch_model, save_path)
            return iterator
    
    print("Trained!")
    if save_path is not None:
        torch.save(network.state_dict(), save_path)

    return iterator
