import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import math


def split(ids, train, val, test):
    # proportions of train, val, test
    assert (train + val + test == 1)
    IDs = np.unique(ids)
    num_ids = len(IDs)

    # priority given to the test/val sets
    test_split = math.ceil(test * num_ids)
    val_split = math.ceil(val * num_ids)
    train_split = num_ids - val_split - test_split

    train = np.where(np.isin(ids, IDs[:train_split]))[0]
    val = np.where(np.isin(ids, IDs[train_split:train_split + val_split]))[0]
    test = np.where(np.isin(ids, IDs[train_split + val_split:]))[0]

    return train, val, test


def Cal_RMSE(loss):
    return math.sqrt(loss) / 2


def train(model, Dataset, optimizer, scheduler=None, batch_size=64, n_epoch=15, output_dir='./logs',
          subtask_weight=0.5, log_name='1'):
    '''
        model: model to train
        optimizer: optimizer to update weights
        scheduler: scheduling learning rate, used when finetuning pretrained models
    '''
    torch.cuda.empty_cache()
    train_indices, val_indices, test_indices = split(Dataset.trainY[:, 0], 0.7, 0.15,
                                                     0.15)  # indices for the training set
    print('create dataloader...')

    train = Subset(Dataset, indices=train_indices)
    val = Subset(Dataset, indices=val_indices)
    test = Subset(Dataset, indices=test_indices)

    train_loader = DataLoader(train, batch_size=batch_size)
    val_loader = DataLoader(val, batch_size=batch_size)
    test_loader = DataLoader(test, batch_size=batch_size)

    if torch.cuda.is_available():
        gpu_id = 0  # Change this to the desired GPU ID if you have multiple GPUs
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)  # Wrap the model with DataParallel
    print("HI")

    model = model.to(device)
    criterion = nn.MSELoss()
    criterion = criterion.to(device)

    # Initialize tensorboard
    writer = SummaryWriter(log_dir=output_dir + '/' + log_name)

    # Initialize lists to store losses
    train_losses = []
    val_losses = []
    test_losses = []
    print('training...')
    
    # Train the model
    for epoch in range(n_epoch):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_position_loss = 0.0
        epoch_train_pupil_loss = 0.0
        epoch_train_reconstruction_loss = 0.0
        for i, (inputs, targets,pupil_size, index) in tqdm(enumerate(train_loader)):
            # Move the inputs and targets to the GPU (if available)
            inputs = inputs.to(device)
            targets = targets.to(device)
            pupil_size = pupil_size.to(device)

            # Compute the outputs and loss for the current batch
            optimizer.zero_grad()
            positions, predict_pupil_size = model(inputs,pupil_size)

            position_loss = criterion(positions.squeeze(), targets.squeeze())
            pupil_size_loss = criterion(predict_pupil_size.squeeze(), pupil_size.squeeze())
            loss = position_loss + pupil_size_loss * subtask_weight
            
            # Compute the gradients and update the parameters
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            epoch_train_position_loss += position_loss.item()
            epoch_train_reconstruction_loss += pupil_size_loss.item()

            # Print the loss and accuracy for the current batch
            if i % 100 == 0:
                print(f"Epoch {epoch}, Batch {i}, position loss: {position_loss.item()}" +
                      f" reconstruction loss: {pupil_size_loss.item()} RMSE(mm): {Cal_RMSE(position_loss.item())}")

        epoch_train_loss /= len(train_loader)
        epoch_train_position_loss /= len(train_loader)
        epoch_train_reconstruction_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)

        # Log
        writer.add_scalar('Loss/train', epoch_train_loss, epoch)
        writer.add_scalar('Position_Loss/train', epoch_train_position_loss, epoch)
        writer.add_scalar('Reconstruction_Loss/train', epoch_train_reconstruction_loss, epoch)
        writer.add_scalar('RMSE of Position Loss/train', Cal_RMSE(epoch_train_position_loss), epoch)

        # Evaluate the model on the validation set
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_position_loss = 0.0
            val_reconstruction_loss = 0.0
            for inputs, targets,pupil_size, index in val_loader:
                            # Move the inputs and targets to the GPU (if available)
                inputs = inputs.to(device)
                targets = targets.to(device)
                pupil_size = pupil_size.to(device)

                # Compute the outputs and loss for the current batch
                positions, predict_pupil_size = model(inputs,pupil_size)

                position_loss = criterion(positions.squeeze(), targets.squeeze())
                pupil_size_loss = criterion(predict_pupil_size.squeeze(), pupil_size.squeeze())
                loss = position_loss + pupil_size_loss * subtask_weight
                val_loss += loss.item()
                val_position_loss += position_loss.item()
                val_reconstruction_loss += pupil_size_loss.item()

            val_loss /= len(val_loader)
            val_position_loss /= len(val_loader)
            val_reconstruction_loss /= len(val_loader)
            val_losses.append(val_loss)

            # Log
            writer.add_scalar('Loss/validation', val_loss, epoch)
            writer.add_scalar('Position_Loss/validation', val_position_loss, epoch)
            writer.add_scalar('Reconstruction_Loss/validation', val_reconstruction_loss, epoch)
            writer.add_scalar('RMSE of Position Loss/validation', Cal_RMSE(val_position_loss), epoch)

            print(f"Epoch {epoch}, Val Loss: {val_loss}, RMSE(mm): {Cal_RMSE(val_position_loss)}")

        with torch.no_grad():
            val_loss = 0.0
            val_position_loss = 0.0
            val_reconstruction_loss = 0.0
            for inputs, targets,pupil_size, index in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                pupil_size = pupil_size.to(device)

                # Compute the outputs and loss for the current batch
                positions, predict_pupil_size = model(inputs,pupil_size)

                position_loss = criterion(positions.squeeze(), targets.squeeze())
                pupil_size_loss = criterion(predict_pupil_size.squeeze(), pupil_size.squeeze())
                loss = position_loss + pupil_size_loss * subtask_weight
                val_loss += loss.item()
                val_position_loss += position_loss.item()
                val_reconstruction_loss += pupil_size_loss.item()

            val_loss /= len(val_loader)
            val_position_loss /= len(val_loader)
            val_reconstruction_loss /= len(val_loader)
            val_losses.append(val_loss)

            # Log
            writer.add_scalar('Loss/test', val_loss, epoch)
            writer.add_scalar('Position_Loss/test', val_position_loss, epoch)
            writer.add_scalar('Reconstruction_Loss/test', val_reconstruction_loss, epoch)
            writer.add_scalar('RMSE of Position Loss/test', Cal_RMSE(val_position_loss), epoch)

            print(f"Epoch {epoch}, test Loss: {val_loss}, RMSE(mm): {Cal_RMSE(val_position_loss)}")

        if scheduler is not None:
            scheduler.step()

    writer.close()
