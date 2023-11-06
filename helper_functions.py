from models.EEGViT_pretrained import EEGViT_pretrained
from models.EEGViT import EEGViT_raw
from models.ViTBase import ViTBase
from models.ViTBase_pretrained import ViTBase_pretrained
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


def train(model, Dataset, optimizer, scheduler=None, batch_size=64, n_epoch=15, output_dir='./logs', log_name='1'):
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

        for i, (inputs, targets, index) in tqdm(enumerate(train_loader)):
            # Move the inputs and targets to the GPU (if available)
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Compute the outputs and loss for the current batch
            optimizer.zero_grad()
            outputs = model(inputs)

            # loss = criterion(outputs.squeeze(), targets.squeeze())
            loss = criterion(outputs.squeeze(), targets.squeeze())

            # Compute the gradients and update the parameters
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

            # Print the loss and accuracy for the current batch
            if i % 100 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()} RMSE(mm): {Cal_RMSE(loss.item())}")

        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)

        # Log
        writer.add_scalar('Loss/train', loss.item(), epoch)
        writer.add_scalar('RMSE of Position Loss/train', Cal_RMSE(loss.item()), epoch)

        # Evaluate the model on the validation set
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for inputs, targets, index in val_loader:
                # Move the inputs and targets to the GPU (if available)
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Compute the outputs and loss for the current batch
                outputs = model(inputs)

                # print(outputs)
                loss = criterion(outputs.squeeze(), targets.squeeze())
                val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            # Log
            writer.add_scalar('Loss/validation', val_loss, epoch)
            writer.add_scalar('RMSE of Position Loss/validation', Cal_RMSE(val_loss), epoch)

            print(f"Epoch {epoch}, Val Loss: {val_loss}, RMSE(mm): {Cal_RMSE(val_loss)}")

        with torch.no_grad():
            val_loss = 0.0
            for inputs, targets, index in test_loader:
                # Move the inputs and targets to the GPU (if available)
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Compute the outputs and loss for the current batch
                outputs = model(inputs)

                loss = criterion(outputs.squeeze(), targets.squeeze())
                val_loss += loss.item()

            val_loss /= len(test_loader)
            test_losses.append(val_loss)

            # Log
            writer.add_scalar('Loss/validation', loss.item(), epoch)
            writer.add_scalar('RMSE/validation', Cal_RMSE(loss.item()), epoch)

            print(f"Epoch {epoch}, test Loss: {val_loss}, RMSE(mm): {Cal_RMSE(val_loss)}")

        if scheduler is not None:
            scheduler.step()

    writer.close()
