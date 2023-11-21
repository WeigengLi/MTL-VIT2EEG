
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import math
from abc import ABC, abstractmethod

LOG_DIR = './logs'
TRAIN_STAGE = 'train'
TEST_STAGE = 'test'
VAL_STAGE = 'val'


class ModelTrainer(ABC):
    def __init__(self, model, Dataset, optimizer, scheduler=None, batch_size=64, n_epoch=15, Trainer_name='Trainer') -> None:
        self.model = model
        self.Dataset = Dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.Trainer_name = Trainer_name
        self.initialization()

    def initialization(self):
        model = self.model
        Dataset = self.Dataset
        batch_size = self.batch_size
        torch.cuda.empty_cache()
        train_indices, val_indices, test_indices = split(Dataset.trainY[:, 0], 0.7, 0.15,
                                                         0.15)  # indices for the training set
        print('create dataloader...')
        train = Subset(Dataset, indices=train_indices)
        val = Subset(Dataset, indices=val_indices)
        test = Subset(Dataset, indices=test_indices)

        self.train_loader = DataLoader(train, batch_size=batch_size)
        self.val_loader = DataLoader(val, batch_size=batch_size)
        self.test_loader = DataLoader(test, batch_size=batch_size)

        if torch.cuda.is_available():
            gpu_id = 0  # Change this to the desired GPU ID if you have multiple GPUs
            torch.cuda.set_device(gpu_id)
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cpu")
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)  # Wrap the model with DataParallel
        print(f"Using Device : {torch.cuda.get_device_name()}")
        model = model.to(device)
        # Initialize tensorboard
        from datetime import datetime
        formatted_date = datetime.now().strftime("%y%m%d")
        self.writer = SummaryWriter(
            log_dir=f'{LOG_DIR}/{formatted_date}/{self.Trainer_name}')
        self.device = device

    def write_logs(self, stage, losses, epoch):
        for key, value in losses.items():
            self.writer.add_scalar(f'{key}/{stage}', value, epoch)

    def run(self):
        model = self.model
        # Initialize lists to store losses
        print('training...')
        # Train the model
        for epoch in range(self.n_epoch):
            # training stage
            model.train()
            loss = self.model_evaluate(TRAIN_STAGE, self.train_loader, epoch)
            self.write_logs(TRAIN_STAGE, loss, epoch)

            # Evaluate the model on the validation set
            model.eval()
            with torch.no_grad():
                # Validate set
                loss = self.model_evaluate(VAL_STAGE, self.val_loader, epoch)
                self.write_logs(VAL_STAGE, loss, epoch)

                # Testset
                loss = self.model_evaluate(TEST_STAGE, self.test_loader, epoch)
                self.write_logs(TEST_STAGE, loss, epoch)

            if self.scheduler is not None:
                self.scheduler.step()
        self.writer.close()

    @abstractmethod
    def model_evaluate(self, stage, data_loader, epoch):
        '''
        Core Evaluate should be same for all stage
        '''
        pass

class MTL_RE_Trainer(ModelTrainer):
    def __init__(self, model, Dataset, optimizer, scheduler=None, batch_size=64, n_epoch=15, Trainer_name='Trainer', weight=0) -> None:
        super().__init__(model, Dataset, optimizer,
                         scheduler, batch_size, n_epoch, Trainer_name)
        self.weight = weight

    def model_evaluate(self, stage, data_loader, epoch):
        device = self.device
        optimizer = self.optimizer
        criterion = nn.MSELoss()
        criterion = criterion.to(self.device)
        epoch_loss = 0.0
        epoch_position_loss = 0.0
        epoch_reconstruction_loss = 0.0
        if stage == TRAIN_STAGE:
            enumerator = tqdm(enumerate(data_loader))
        else:
            enumerator = enumerate(data_loader)
        for i, (inputs, targets, *index) in enumerator:
           
            # Move the inputs and targets to the GPU (if available)
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Compute the outputs and loss for the current batch
            self.optimizer.zero_grad()
            positions, x_reconstructed = self.model(inputs)

            position_loss = criterion(positions.squeeze(), targets.squeeze())
            reconstruction_loss = criterion(
                x_reconstructed.squeeze(), inputs.squeeze())
            loss = position_loss + self.weight * reconstruction_loss
            # Compute the gradients and update the parameters
            if stage == TRAIN_STAGE:
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item()
            epoch_position_loss += position_loss.item()
            epoch_reconstruction_loss += reconstruction_loss.item()

            # Print the loss and accuracy for the current batch
            if stage == TRAIN_STAGE and i % 100 == 0:
                print(f"Epoch {epoch}, Batch {i}, position loss: {position_loss.item()}" +
                      f" reconstruction loss: {default_round(reconstruction_loss.item())} RMSE(mm): {default_round(Cal_RMSE(position_loss.item()))}")

        loss = {'overall_loss': epoch_loss / len(data_loader),
                'position_loss': epoch_position_loss / len(data_loader),
                'position_RMSE': Cal_RMSE(epoch_position_loss / len(data_loader)),
                'reconstruction_loss': epoch_reconstruction_loss / len(data_loader)
                }
        if stage in [TEST_STAGE, VAL_STAGE]:
            print(
                f"Epoch {epoch}, {stage} Loss: {default_round(loss['overall_loss'])}, RMSE(mm): {default_round(loss['position_RMSE'])}")
        return loss

class MTL_PU_Trainer(ModelTrainer):
    def __init__(self, model, Dataset, optimizer, scheduler=None, batch_size=64, n_epoch=15, Trainer_name='Trainer', weight=0) -> None:
        super().__init__(model, Dataset, optimizer,
                         scheduler, batch_size, n_epoch, Trainer_name)
        self.weight = weight

    def model_evaluate(self, stage, data_loader, epoch):
        device = self.device
        optimizer = self.optimizer
        criterion = nn.MSELoss()
        criterion = criterion.to(self.device)
        epoch_loss = 0.0
        epoch_position_loss = 0.0
        epoch_pupil_loss = 0.0
        if stage == TRAIN_STAGE:
            enumerator = tqdm(enumerate(data_loader))
        else:
            enumerator = enumerate(data_loader)
        for i, (inputs, targets, pupil_size, *index) in enumerator:
            # Move the inputs and targets to the GPU (if available)
            inputs = inputs.to(device)
            targets = targets.to(device)
            pupil_size = pupil_size.to(device)

            # Compute the outputs and loss for the current batch
            optimizer.zero_grad()
            positions, predict_size = self.model(inputs, pupil_size)

            position_loss = criterion(positions.squeeze(), targets.squeeze())
            pupil_size_loss = criterion(
                predict_size.squeeze(), pupil_size.squeeze())
            loss = position_loss + pupil_size_loss * self.weight

            # Compute the gradients and update the parameters
            if stage == TRAIN_STAGE:
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item()
            epoch_position_loss += position_loss.item()
            epoch_pupil_loss += pupil_size_loss.item()
            
            # Print the loss and accuracy for the current batch
            if stage == TRAIN_STAGE and i % 100 == 0:
                print(f"Epoch {epoch}, Batch {i}, position loss: {position_loss.item()}" +
                      f" reconstruction loss: {default_round(pupil_size_loss.item())} RMSE(mm): {default_round(Cal_RMSE(position_loss.item()))}")

        loss = {'overall_loss': epoch_loss / len(data_loader),
                'position_loss': epoch_position_loss / len(data_loader),
                'position_RMSE': Cal_RMSE(epoch_position_loss / len(data_loader)),
                'pupil_size_loss': epoch_pupil_loss / len(data_loader)
                }
        if stage in [TEST_STAGE, VAL_STAGE]:
            print(
                f"Epoch {epoch}, {stage} Loss: {default_round(loss['overall_loss'])}, RMSE(mm): {default_round(loss['position_RMSE'])}")
        return loss

class STL_Trainer(ModelTrainer):
    def __init__(self, model, Dataset, optimizer, scheduler=None, batch_size=64, n_epoch=15, Trainer_name='Trainer') -> None:
        super().__init__(model, Dataset, optimizer, scheduler, batch_size, n_epoch, Trainer_name)
        
    def model_evaluate(self, stage, data_loader, epoch):
        device = self.device
        optimizer = self.optimizer
        criterion = nn.MSELoss()
        criterion = criterion.to(self.device)
        epoch_loss = 0.0
        epoch_position_loss = 0.0
        
        enumerator = tqdm(enumerate(data_loader)) if stage == TRAIN_STAGE else enumerate(data_loader)
       
        for i, (inputs, targets, *index) in enumerator:
            # Move the inputs and targets to the GPU (if available)
            inputs = inputs.to(device)
            targets = targets.to(device)
            # Compute the outputs and loss for the current batch
            if stage == TRAIN_STAGE:
                optimizer.zero_grad()
            outputs = self.model(inputs)

            # loss = criterion(outputs.squeeze(), targets.squeeze())
            position_loss = criterion(outputs.squeeze(), targets.squeeze())
            # Compute the gradients and update the parameters
            if stage == TRAIN_STAGE:
                position_loss.backward()
                optimizer.step()
            epoch_loss += position_loss.item()
            epoch_position_loss += position_loss.item()

            
            # Print the loss and accuracy for the current batch
            if stage == TRAIN_STAGE and i % 100 == 0:
                print(f"Epoch {epoch}, Batch {i}, position loss: {position_loss.item()}" +
                      f" RMSE(mm): {default_round(Cal_RMSE(position_loss.item()))}")

        loss = {'overall_loss': epoch_loss / len(data_loader),
                'position_loss': epoch_position_loss / len(data_loader),
                'position_RMSE': Cal_RMSE(epoch_position_loss / len(data_loader)),
                }
        if stage in [TEST_STAGE, VAL_STAGE]:
            print(
                f"Epoch {epoch}, {stage} Loss: {default_round(loss['overall_loss'])}, RMSE(mm): {default_round(loss['position_RMSE'])}")
        return loss

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

def default_round(a):
    return round(a, 2)

