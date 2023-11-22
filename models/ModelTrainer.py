
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import math
from abc import ABC, abstractmethod
from itertools import cycle

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
        self.criterion = nn.MSELoss()
        self.criterion.to(self.device)

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
        criterion = self.criterion
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
            if stage == TRAIN_STAGE:
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
        criterion = self.criterion
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
            if stage == TRAIN_STAGE:
                self.optimizer.zero_grad()
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
        super().__init__(model, Dataset, optimizer,
                         scheduler, batch_size, n_epoch, Trainer_name)

    def model_evaluate(self, stage, data_loader, epoch):
        device = self.device
        optimizer = self.optimizer

        epoch_loss = 0.0
        epoch_position_loss = 0.0

        enumerator = tqdm(enumerate(data_loader)
                          ) if stage == TRAIN_STAGE else enumerate(data_loader)
        criterion = self.criterion
        for i, (inputs, targets, *index) in enumerator:
            # Move the inputs and targets to the GPU (if available)
            inputs = inputs.to(device)
            targets = targets.to(device)
            # Compute the outputs and loss for the current batch
            if stage == TRAIN_STAGE:
                self.optimizer.zero_grad()
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


class MTL_ADDA_Trainer(ModelTrainer):
    def __init__(self, model, Dataset, optimizer, discriminator, scheduler=None, batch_size=64, n_epoch=15, weight=1, Trainer_name='Trainer') -> None:
        self.discriminator = discriminator
        self.weight = weight
        super().__init__(model, Dataset, optimizer,
                         scheduler, batch_size, n_epoch, Trainer_name)

    def initialization(self):
        super().initialization()
        self.optimizer = torch.optim.Adam(
            list(self.discriminator.parameters()) + list(self.model.parameters()), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=6, gamma=0.1)
        self.discriminator.to(self.device)
        self.BCE_criterion = nn.BCEWithLogitsLoss()
        self.BCE_criterion.to(self.device)
        # 计算复制 target_loader 的倍数，以使其长度不小于 source_loader

    def model_evaluate(self, stage, data_loader, epoch):
        device = self.device
        optimizer = self.optimizer

        MSE_criterion = self.criterion
        BCE_criterion = self.BCE_criterion
        epoch_loss = 0.0
        epoch_position_loss = 0.0
        epoch_domain_loss = 0.0
        if stage == TRAIN_STAGE:
            source_loader = self.train_loader
            target_loader = self.test_loader
            c_target_loader = cycle(target_loader)
            batches = zip(source_loader,  cycle(target_loader))
            first_batch = next(batches)
            n_batches = len(source_loader)
            for i, ((source_x, source_labels, index), (target_x, trage_y, index2)) in tqdm(enumerate(batches), total=n_batches):
                optimizer.zero_grad()
                x = torch.cat([source_x, target_x])
                x = x.to(device)
                domain_y = torch.cat([torch.ones(source_x.shape[0]),
                                      torch.zeros(target_x.shape[0])])
                domain_y = domain_y.to(device)
                label_y = source_labels.to(device)

                positions, shared_features = self.model(x)
                domain_preds = self.discriminator(shared_features).squeeze()
                label_preds = positions[:source_x.shape[0]]

                domain_loss = BCE_criterion(domain_preds, domain_y)
                position_loss = MSE_criterion(label_preds, label_y)
                loss = domain_loss*self.weight + position_loss

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_position_loss += position_loss.item()
                epoch_domain_loss += domain_loss.item()
                # Print the loss and accuracy for the current batch
                if stage == TRAIN_STAGE and i % 100 == 0:
                    print(f"Epoch {epoch}, Batch {i}, position loss: {position_loss.item()} " +
                          f" RMSE(mm): {default_round(Cal_RMSE(position_loss.item()))} " +
                          f" domain loss: {domain_loss.item()}")

            loss = {'overall_loss': epoch_loss / len(data_loader),
                    'position_loss': epoch_position_loss / len(data_loader),
                    'position_RMSE': Cal_RMSE(epoch_position_loss / len(data_loader)),
                    'domain_loss': domain_loss / len(data_loader)
                    }

            return loss

        # Test and Val stage is the same as Single Task Learning
        else:
            enumerator = tqdm(enumerate(data_loader)
                              ) if stage == TRAIN_STAGE else enumerate(data_loader)

            for i, (inputs, targets, *index) in enumerator:
                # Move the inputs and targets to the GPU (if available)
                inputs = inputs.to(device)
                targets = targets.to(device)
                # Compute the outputs and loss for the current batch
                positions, shared_features = self.model(inputs)
                # loss = criterion(outputs.squeeze(), targets.squeeze())
                position_loss = MSE_criterion(
                    positions.squeeze(), targets.squeeze())
                # Compute the gradients and update the parameters
                epoch_loss += position_loss.item()
                epoch_position_loss += position_loss.item()

            loss = {'overall_loss': epoch_loss / len(data_loader),
                    'position_loss': epoch_position_loss / len(data_loader),
                    'position_RMSE': Cal_RMSE(epoch_position_loss / len(data_loader)),
                    }
            if stage in [TEST_STAGE, VAL_STAGE]:
                print(
                    f"Epoch {epoch}, {stage} Loss: {default_round(loss['overall_loss'])}, RMSE(mm): {default_round(loss['position_RMSE'])}")
            return loss


class MTL_ADDA_Trainer2(ModelTrainer):
    def __init__(self, model, Dataset, optimizer, scheduler=None, batch_size=64, n_epoch=15, weight=1, Trainer_name='Trainer') -> None:

        self.weight = weight
        super().__init__(model, Dataset, optimizer,
                         scheduler, batch_size, n_epoch, Trainer_name)

    def initialization(self):
        super().initialization()
        self.BCE_criterion = nn.BCEWithLogitsLoss()
        self.BCE_criterion.to(self.device)
        # 计算复制 target_loader 的倍数，以使其长度不小于 source_loader

    def model_evaluate(self, stage, data_loader, epoch):
        device = self.device
        optimizer = self.optimizer

        MSE_criterion = self.criterion
        BCE_criterion = self.BCE_criterion
        epoch_loss = 0.0
        epoch_position_loss = 0.0
        epoch_domain_loss = 0.0
        if stage == TRAIN_STAGE:
            source_loader = self.train_loader
            target_loader = self.test_loader
            batches = zip(source_loader,  cycle(target_loader))
            n_batches = len(source_loader)
            for i, ((source_x, source_labels, index), (target_x, trage_y, index2)) in tqdm(enumerate(batches), total=n_batches):
                optimizer.zero_grad()
                x = torch.cat([source_x, target_x])
                x = x.to(device)
                domain_y = torch.cat([torch.ones(source_x.shape[0]),
                                      torch.zeros(target_x.shape[0])])
                domain_y = domain_y.to(device)
                label_y = source_labels.to(device)
                positions, domain_preds = self.model(x)

                label_preds = positions[:source_x.shape[0]]
                domain_loss = BCE_criterion(domain_preds.squeeze(), domain_y)
                position_loss = MSE_criterion(label_preds, label_y)
                loss = domain_loss*self.weight + position_loss
                # TODO: Domian loss计算很慢
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_position_loss += position_loss.item()
                epoch_domain_loss += domain_loss.item()
                # Print the loss and accuracy for the current batch
                if stage == TRAIN_STAGE and i % 100 == 0:
                    print(f"Epoch {epoch}, Batch {i}, position loss: {position_loss.item()} " +
                          f" RMSE(mm): {default_round(Cal_RMSE(position_loss.item()))} " +
                          f" domain loss: {domain_loss.item()}")

            loss = {'overall_loss': epoch_loss / len(data_loader),
                    'position_loss': epoch_position_loss / len(data_loader),
                    'position_RMSE': Cal_RMSE(epoch_position_loss / len(data_loader)),
                    'domain_loss': domain_loss / len(data_loader)
                    }

            return loss

        # Test and Val stage is the same as Single Task Learning
        else:
            enumerator = tqdm(enumerate(data_loader)
                              ) if stage == TRAIN_STAGE else enumerate(data_loader)

            for i, (inputs, targets, *index) in enumerator:
                # Move the inputs and targets to the GPU (if available)
                inputs = inputs.to(device)
                targets = targets.to(device)
                # Compute the outputs and loss for the current batch
                positions, shared_features = self.model(inputs)
                # loss = criterion(outputs.squeeze(), targets.squeeze())
                position_loss = MSE_criterion(
                    positions.squeeze(), targets.squeeze())
                # Compute the gradients and update the parameters
                epoch_loss += position_loss.item()
                epoch_position_loss += position_loss.item()

            loss = {'overall_loss': epoch_loss / len(data_loader),
                    'position_loss': epoch_position_loss / len(data_loader),
                    'position_RMSE': Cal_RMSE(epoch_position_loss / len(data_loader)),
                    }
            if stage in [TEST_STAGE, VAL_STAGE]:
                print(
                    f"Epoch {epoch}, {stage} Loss: {default_round(loss['overall_loss'])}, RMSE(mm): {default_round(loss['position_RMSE'])}")
            return loss


class MTL_ADDA_Trainer3(ModelTrainer):
    def __init__(self, model, Dataset, optimizer, discriminator, scheduler=None, batch_size=64, n_epoch=15, weight=1, Trainer_name='Trainer') -> None:
        self.discriminator = discriminator
        self.weight = weight
        super().__init__(model, Dataset, optimizer,
                         scheduler, batch_size, n_epoch, Trainer_name)

    def initialization(self):
        super().initialization()
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=1e-4)
        self.discriminator_scheduler = torch.optim.lr_scheduler.StepLR(
            self.discriminator_optimizer, step_size=6, gamma=0.1)
        self.discriminator.to(self.device)
        self.BCE_criterion = nn.BCEWithLogitsLoss()
        self.BCE_criterion.to(self.device)
        # 计算复制 target_loader 的倍数，以使其长度不小于 source_loader

    def model_evaluate(self, stage, data_loader, epoch):
        device = self.device
        optimizer = self.optimizer

        MSE_criterion = self.criterion
        BCE_criterion = self.BCE_criterion
        epoch_loss = 0.0
        epoch_position_loss = 0.0
        epoch_domain_loss = 0.0
        if stage == TRAIN_STAGE:
            source_loader = self.train_loader
            target_loader = self.test_loader
            c_target_loader = cycle(target_loader)
            batches = zip(source_loader,  cycle(target_loader))
            first_batch = next(batches)
            n_batches = len(source_loader)
            for i, ((source_x, source_labels, index), (target_x, trage_y, index2)) in tqdm(enumerate(batches), total=n_batches):

                x = torch.cat([source_x, target_x])
                x = x.to(device)
                domain_y = torch.cat([torch.ones(source_x.shape[0]),
                                      torch.zeros(target_x.shape[0])])
                domain_y = domain_y.to(device)
                label_y = source_labels.to(device)

                positions, shared_features = self.model(x)
                domain_preds = self.discriminator(shared_features).squeeze()
                label_preds = positions[:source_x.shape[0]]

                domain_loss = BCE_criterion(domain_preds, domain_y)
                position_loss = MSE_criterion(label_preds, label_y)
                self.discriminator_optimizer.zero_grad()
                domain_loss.backward(retain_graph=True)
                self.discriminator_optimizer.step()
                optimizer.zero_grad()
                loss = position_loss-domain_loss.item()*self.weight
                loss.backward()
                self.discriminator_optimizer.step()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_position_loss += position_loss.item()
                epoch_domain_loss += domain_loss.item()
                # Print the loss and accuracy for the current batch
                if stage == TRAIN_STAGE and i % 100 == 0:
                    print(f"Epoch {epoch}, Batch {i}, position loss: {position_loss.item()} " +
                          f" RMSE(mm): {default_round(Cal_RMSE(position_loss.item()))} " +
                          f" domain loss: {domain_loss.item()}")

            loss = {'overall_loss': epoch_loss / len(data_loader),
                    'position_loss': epoch_position_loss / len(data_loader),
                    'position_RMSE': Cal_RMSE(epoch_position_loss / len(data_loader)),
                    'domain_loss': domain_loss / len(data_loader)
                    }

            return loss

        # Test and Val stage is the same as Single Task Learning
        else:
            enumerator = tqdm(enumerate(data_loader)
                              ) if stage == TRAIN_STAGE else enumerate(data_loader)

            for i, (inputs, targets, *index) in enumerator:
                # Move the inputs and targets to the GPU (if available)
                inputs = inputs.to(device)
                targets = targets.to(device)
                # Compute the outputs and loss for the current batch
                positions, shared_features = self.model(inputs)
                # loss = criterion(outputs.squeeze(), targets.squeeze())
                position_loss = MSE_criterion(
                    positions.squeeze(), targets.squeeze())
                # Compute the gradients and update the parameters
                epoch_loss += position_loss.item()
                epoch_position_loss += position_loss.item()

            loss = {'overall_loss': epoch_loss / len(data_loader),
                    'position_loss': epoch_position_loss / len(data_loader),
                    'position_RMSE': Cal_RMSE(epoch_position_loss / len(data_loader)),
                    }
            if stage in [TEST_STAGE, VAL_STAGE]:
                print(
                    f"Epoch {epoch}, {stage} Loss: {default_round(loss['overall_loss'])}, RMSE(mm): {default_round(loss['position_RMSE'])}")
            return loss


class MTL_ADDA_Trainer4(ModelTrainer):
    def __init__(self, model, Dataset, optimizer, discriminator, pretrained_model=None, scheduler=None, batch_size=64, n_epoch=15, weight=1, Trainer_name='Trainer') -> None:
        self.discriminator = discriminator
        self.weight = weight
        self.pretrained_model = pretrained_model
        super().__init__(model, Dataset, optimizer,
                         scheduler, batch_size, n_epoch, Trainer_name)
        self.STL_Trainer = STL_Trainer(model, Dataset, optimizer,
                                       scheduler, batch_size, n_epoch, Trainer_name)

    def initialization(self):
        super().initialization()
        self.discriminator_optimizer = torch.optim.Adam(
            list(self.discriminator.parameters())+list(self.model.parameters()), lr=1e-4)
        self.discriminator_scheduler = torch.optim.lr_scheduler.StepLR(
            self.discriminator_optimizer, step_size=6, gamma=0.1)
        self.discriminator.to(self.device)
        self.pretrained_model.to(self.device)
        self.pretrained_model.eval()
        self.BCE_criterion = nn.BCEWithLogitsLoss()
        self.BCE_criterion.to(self.device)
        # 计算复制 target_loader 的倍数，以使其长度不小于 source_loader

    def model_evaluate(self, stage, data_loader, epoch):

        if stage == TRAIN_STAGE:
            device = self.device
            optimizer = self.optimizer

            BCE_criterion = self.BCE_criterion
            epoch_loss = 0.0
            epoch_position_loss = 0.0
            epoch_domain_loss = 0.0
            source_loader = self.train_loader
            target_loader = self.test_loader
            batches = zip(cycle(source_loader),  cycle(target_loader))
            n_batches = len(target_loader)
            for i, ((source_x, source_labels, index), (target_x, trage_y, index2)) in tqdm(enumerate(batches), total=n_batches):

                source_x.to(device)
                target_x.to(device)
                domain_y = torch.cat([torch.ones(source_x.shape[0]),
                                      torch.zeros(target_x.shape[0])])
                domain_y = domain_y.to(device)
                with torch.no_grad():
                    positions, source_shared_features = self.pretrained_model(
                        source_x)
                positions, targe_shared_features = self.model(target_x)
                domain_preds = self.discriminator(
                    torch.cat(source_shared_features, targe_shared_features)).squeeze()
                domain_loss = BCE_criterion(domain_preds, domain_y)
                domain_loss.backward(retain_graph=True)
                self.discriminator_optimizer.zero_grad()
                self.discriminator_optimizer.step()
                optimizer.zero_grad()
                model_loss = -domain_loss
                model_loss.backward()
                optimizer.step()

                epoch_domain_loss += domain_loss.item()
                # Print the loss and accuracy for the current batch
                if stage == TRAIN_STAGE and i % 100 == 0:
                    print(f" domain loss: {domain_loss.item()}")

            loss = {
                'domain_loss': domain_loss / len(data_loader)
            }

            return loss

        # Test and Val stage is the same as Single Task Learning
        else:
            device = self.device
            optimizer = self.optimizer

            epoch_loss = 0.0
            epoch_position_loss = 0.0

            enumerator = tqdm(enumerate(data_loader)
                              ) if stage == TRAIN_STAGE else enumerate(data_loader)
            criterion = self.criterion
            for i, (inputs, targets, *index) in enumerator:
                # Move the inputs and targets to the GPU (if available)
                inputs = inputs.to(device)
                targets = targets.to(device)
                # Compute the outputs and loss for the current batch
                if stage == TRAIN_STAGE:
                    self.optimizer.zero_grad()
                outputs, sf = self.model(inputs)

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


class MTL_ADDA_Trainer_with_pre(ModelTrainer):
    def __init__(self, model, Dataset, optimizer, discriminator, pretrained_model=None, scheduler=None, batch_size=64, n_epoch=15, weight=1, Trainer_name='Trainer') -> None:
        self.discriminator = discriminator
        self.weight = weight
        self.pretrained_model = pretrained_model
        super().__init__(model, Dataset, optimizer,
                         scheduler, batch_size, n_epoch, Trainer_name)

    def initialization(self):
        super().initialization()
        self.discriminator_optimizer = torch.optim.Adam(
            list(self.discriminator.parameters())+list(self.model.parameters()), lr=1e-4)
        self.discriminator_scheduler = torch.optim.lr_scheduler.StepLR(
            self.discriminator_optimizer, step_size=6, gamma=0.1)
        self.discriminator.to(self.device)
        self.pretrained_model.to(self.device)
        self.pretrained_model.eval()
        self.BCE_criterion = nn.BCEWithLogitsLoss()
        self.BCE_criterion.to(self.device)
        # 计算复制 target_loader 的倍数，以使其长度不小于 source_loader

    def STL_model_evaluate(self, stage, data_loader, epoch):
        device = self.device
        optimizer = self.optimizer
        epoch_loss = 0.0
        epoch_position_loss = 0.0

        enumerator = tqdm(enumerate(data_loader)
                          ) if stage == TRAIN_STAGE else enumerate(data_loader)
        criterion = self.criterion
        for i, (inputs, targets, *index) in enumerator:
            # Move the inputs and targets to the GPU (if available)
            inputs = inputs.to(device)
            targets = targets.to(device)
            # Compute the outputs and loss for the current batch
            if stage == TRAIN_STAGE:
                self.optimizer.zero_grad()
            outputs, sf = self.model(inputs)

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

    def adda_model_evaluate(self, stage, data_loader, epoch):
        if stage == TRAIN_STAGE:
            with torch.no_grad():
                self.STL_model_evaluate(TEST_STAGE, self.test_loader, epoch)
            device = self.device
            optimizer = self.optimizer

            BCE_criterion = self.BCE_criterion

            epoch_domain_loss = 0.0
            source_loader = self.train_loader
            target_loader = self.test_loader
            batches = zip(cycle(source_loader),  cycle(target_loader))
            n_batches = len(target_loader)
            for i, ((source_x, source_labels, index), (target_x, trage_y, index2)) in tqdm(enumerate(batches), total=n_batches):

                if i == n_batches: break
                source_x = source_x.to(device)
                target_x = target_x.to(device)
                source_y = torch.ones(source_x.shape[0])
                targe_y = torch.zeros(target_x.shape[0])
                targe_y = targe_y.to(device)
                source_y = source_y.to(device)
                domain_y = torch.cat([source_y, targe_y])
                domain_y = domain_y.to(device)
                with torch.no_grad():
                    positions, source_shared_features = self.pretrained_model(
                        source_x)
                positions, targe_shared_features = self.model(target_x)
                source_preds = self.discriminator(
                    source_shared_features).squeeze()
                target_preds = self.discriminator(targe_shared_features).squeeze()

               
                
                optimizer.zero_grad()
                target_loss = BCE_criterion(target_preds, targe_y)
                copied_target_loss = target_loss.clone().detach()
                model_loss = -target_loss
                model_loss.backward()
                optimizer.step()
                
                self.discriminator_optimizer.zero_grad()
                source_loss = BCE_criterion(source_preds, source_y)
                domain_loss = source_loss + copied_target_loss
                domain_loss.backward()
                self.discriminator_optimizer.step()
                


                epoch_domain_loss += domain_loss.item()
                # Print the loss and accuracy for the current batch
                if stage == TRAIN_STAGE and i % 25 == 0:
                    print(f" domain loss: {domain_loss.item()}")

            loss = {'domain_loss': domain_loss / len(data_loader)}
            return loss
        # Test and Val stage is the same as Single Task Learning
        else:
            return self.STL_model_evaluate(stage, data_loader, epoch)

    def pretrain_model(self):
        self.model_evaluate = self.STL_model_evaluate
        self.run()

    def model_evaluate(self, stage, data_loader, epoch):
        pass

    def train_adda(self):

        self.model_evaluate = self.adda_model_evaluate
        self.run()


# region helper functions

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

# endregion
