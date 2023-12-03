
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from copy import deepcopy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import math
from abc import ABC, abstractmethod
from itertools import cycle
import math
import pandas as pd
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
LOG_DIR = './logs'
TRAIN_STAGE = 'train'
TEST_STAGE = 'test'
VAL_STAGE = 'val'


class ModelTrainer(ABC):
    def __init__(self, model, Dataset, optimizer, scheduler=None, batch_size=64, n_epoch=15, Trainer_name='Trainer', weight = 0) -> None:
        self.model = model
        self.Dataset = Dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.Trainer_name = Trainer_name
        self.weight = weight
        self.plots = {}
        self.initialization()

    def initialization(self, drop_last=False):
        '''
        defin a standard initialization for all models
        if you want to send
        args:
            drop_last: bool, whether to drop the last batch in the dataloader
                        this will make sure all batches have the same size

        '''
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

        self.train_loader = DataLoader(
            train, batch_size=batch_size, drop_last=drop_last)
        self.val_loader = DataLoader(
            val, batch_size=batch_size, drop_last=drop_last)
        self.test_loader = DataLoader(
            test, batch_size=batch_size, drop_last=drop_last)

        if torch.cuda.is_available():
            # Set the random seed
            torch.manual_seed(0)
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
        '''
        Define a standard train/val/test loop  for models
        '''
        model = self.model
        # Initialize lists to store losses
        print('training...')
        # Train the model
        for epoch in range(self.n_epoch):
            # training stage
            model.train()
            self.clear_plots()
            loss = self.model_evaluate(TRAIN_STAGE, self.train_loader, epoch)
            self.write_logs(TRAIN_STAGE, loss, epoch)
            # Evaluate the model on the validation set
            model.eval()
            with torch.no_grad():
                # Validate set
                loss = self.model_evaluate(VAL_STAGE, self.val_loader, epoch)
                self.write_logs(VAL_STAGE, loss, epoch)
                print(f"Epoch {epoch}, {TEST_STAGE} Loss: {default_round(loss['overall_loss'])}, RMSE(mm): {default_round(loss['position_RMSE'])}")
                # Testset
                loss = self.model_evaluate(TEST_STAGE, self.test_loader, epoch)
                self.write_logs(TEST_STAGE, loss, epoch)

                print(f"Epoch {epoch}, {TEST_STAGE} Loss: {default_round(loss['overall_loss'])}, RMSE(mm): {default_round(loss['position_RMSE'])}")
            self.plot(f'Epoch_{epoch}.html')
            if self.scheduler is not None:
                self.scheduler.step()
        self.writer.close()

    def Sandard_model_evaluate(self, stage, data_loader, epoch, ):
        '''
        Define a standard evaluate function for all test stages
        args:
            stage: str, 'train', 'test', 'val'
            data_loader: torch.utils.data.DataLoader 
            epoch: int, current epoch
        return:
            loss: dict, b  
                e.g. {'overall_loss': float, 'position_loss': float, 'position_RMSE': float}

        '''
        device = self.device
        optimizer = self.optimizer
        epoch_loss = 0.0
        epoch_position_loss = 0.0

        enumerator = tqdm(enumerate(data_loader)
                            ) if stage == TRAIN_STAGE else enumerate(data_loader)
        criterion = self.criterion
        epoch_pred_position = torch.tensor([]).to(self.device)
        epoch_lables = torch.tensor([]).to(self.device)
        for i, (inputs, targets, pupil_size, *index) in enumerator:
            # Move the inputs and targets to the GPU (if available)
            inputs = inputs.to(device)
            targets = targets.to(device)
            # Compute the outputs and loss for the current batch
            if stage == TRAIN_STAGE:
                self.optimizer.zero_grad()
            outputs, *sf = self.model(inputs)

            # loss = criterion(outputs.squeeze(), targets.squeeze())
            position_loss = criterion(outputs.squeeze(), targets.squeeze())
            # Compute the gradients and update the parameters
            if stage == TRAIN_STAGE:
                position_loss.backward()
                optimizer.step()
            epoch_loss += position_loss.item()
            epoch_position_loss += position_loss.item()
            epoch_pred_position = torch.cat([epoch_pred_position, outputs])
            epoch_lables = torch.cat([epoch_lables, targets])
            # Print the loss and accuracy for the current batch
            if stage == TRAIN_STAGE and i % 100 == 0:
                print(f"Epoch {epoch}, Batch {i}, position loss: {position_loss.item()}" +
                        f" RMSE(mm): {default_round(Cal_RMSE(position_loss.item()))}")

        loss = {'overall_loss': epoch_loss / len(data_loader),
                'position_loss': epoch_position_loss / len(data_loader),
                'position_RMSE': Cal_RMSE(epoch_position_loss / len(data_loader)),
                }
        if stage == VAL_STAGE:
            self.val_results = epoch_pred_position
            self.val_lables = epoch_lables
        if stage == TEST_STAGE:
            self.test_results = epoch_pred_position
            self.test_lables = epoch_lables
            save_path = f'logs/{self.Trainer_name}/predict_position{epoch}.html'
            self.plot_positions(self, save_path)

        if stage in [TEST_STAGE, VAL_STAGE]:
            print(
                f"Epoch {epoch}, {stage} Loss: {default_round(loss['overall_loss'])}, RMSE(mm): {default_round(loss['position_RMSE'])}")
        return loss

    def to_device(self, tensors, device=None):
        '''
        Send them all to device
        args:
            tensors: list of torch.Tensor
        '''
        device = self.device if device is None else device

        return [tensor.to(device) for tensor in tensors]

    # Plotting functions

    def clear_plots(self):
        self.plots = {}

    def save_to_plot_elements(self, plot_name, ele_dict):
        '''
        save torch elements to plot_elements
        '''
        if plot_name not in self.plots.keys():
            self.plots[plot_name] = {}
        plot_dict = self.plots[plot_name]
        for key in ele_dict.keys():
            if key not in plot_dict:
                plot_dict[key] = ele_dict[key]
            else:
                plot_dict[key] = torch.cat([plot_dict[key], ele_dict[key]])

    def plot(self, file_name):
        '''
        a function to plot all plots in self.plots
        args:
            plot_name: str, the name of the plot
        '''
        if not self.plots:
            return
        from datetime import datetime
        formatted_date = datetime.now().strftime("%y%m%d")
        for plot_name, plot_eles in self.plots.items():
            plot_save_dir = os.path.join(LOG_DIR,formatted_date,self.Trainer_name, plot_name)
            os.makedirs(plot_save_dir, exist_ok=True)
            plot_save_path = os.path.join(plot_save_dir,file_name)
            label_mapping = {}
            eles, lables = self.to_device([torch.tensor([]), torch.tensor([])])
            for i, (key, values) in enumerate(plot_eles.items()):
                eles = torch.cat([eles, values])
                lables = torch.cat([lables, torch.ones(values.shape[0]).to(self.device)*i])
                label_mapping[i] = key
            eles = eles.cpu().detach().numpy()
            lables = lables.cpu().detach().numpy()
            if eles.shape[1] == 2:
                df = pd.DataFrame(data=eles, columns=['x', 'y'])
            elif eles.shape[1] == 3:
                df = pd.DataFrame(data=eles, columns=['x', 'y', 'z'])   
            else:
                scaler = StandardScaler()
                eles_normalized = scaler.fit_transform(eles)
                # Apply PCA
                n_components = 3
                pca = PCA(n_components=n_components)
                eles_pca = pca.fit_transform(eles_normalized)
                df = pd.DataFrame(data=eles_pca, columns=['PC1', 'PC2', 'PC3'])
                df['Labels'] = lables
                explained_var_ratio = pca.explained_variance_ratio_
                info_percentage = [
                    f"{round(ratio * 100, 2)}%" for ratio in explained_var_ratio]
            df['Labels'] = lables
            # 使用 map 方法应用映射
            df['Labels'] = df['Labels'].map(label_mapping)
            # 将 'Domain Labels' 列转换为分类数据类型
            df['Labels'] = df['Labels'].astype('category')

            if eles.shape[1] == 2:
                # Plotting with Plotly
                fig = px.scatter(df, x='x', y='y',
                                 color='Labels', symbol='Labels')

            elif eles.shape[1] == 3:
                fig = px.scatter(df, x='x', y='y', z='z',
                                 color='Labels', symbol='Labels')   
            else:
                # Plotting with Plotly
                fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', color='Labels', symbol='Labels',
                                    labels={'PC1': f'Principal Component 1 ({info_percentage[0]})',
                                            'PC2': f'Principal Component 2 ({info_percentage[1]})',
                                            'PC3': f'Principal Component 3 ({info_percentage[2]})'},
                                    title='PCA of Normalized Shear Features with Domain Labels')
            fig.write_html(plot_save_path)


class MTL_RE_Trainer(ModelTrainer):
    def __init__(self, model, Dataset, optimizer, scheduler=None, batch_size=64, n_epoch=15,
                 Trainer_name='Trainer', weight=0) -> None:
        super().__init__(model, Dataset, optimizer, scheduler, batch_size, 
                         n_epoch, Trainer_name, weight=weight,)

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
            positions, x_reconstructed, sf = self.model(inputs)
            self.save_to_plot_elements('positions', {f'{stage}_predict_positions': positions ,
                                                     f'{stage}_lables': targets})      
            self.save_to_plot_elements('shear_feature', {f'{stage}_shear_feature': sf })                        
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
            positions, predict_size, *sf = self.model(inputs)
            self.save_to_plot_elements('positions', {f'{stage}_predict_positions_pupuil_size': positions ,
                                                     f'{stage}_lables': targets})
            self.save_to_plot_elements('positions', {f'{stage}_predict_positions_pred_pupuil_size': torch.cat([positions, predict_size], dim=1) ,
                                                     f'{stage}_lables': targets})
            self.save_to_plot_elements('positions', {f'{stage}_predict_positions_real_pupuil_size': torch.cat([positions, pupil_size], dim=1) ,
                                                     f'{stage}_lables': targets})
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
                optimizer.zero_grad()
            outputs = self.model(inputs)
            self.save_to_plot_elements('positions', {f'{stage}_predict_positions': outputs ,
                                                     f'{stage}_lables': targets})
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



# model和 classifier相同模型和计算图
# 训练缓慢4.13s/it
class MTL_ADDA_Trainer_with_dis(ModelTrainer):
    def __init__(self, model, Dataset, optimizer, scheduler=None, batch_size=64, n_epoch=15, weight=1, Trainer_name='Trainer') -> None:
        self.weight = weight
        super().__init__(model, Dataset, optimizer,
                         scheduler, batch_size, n_epoch, Trainer_name)

    def initialization(self):
        super().initialization()
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

    def model_evaluate(self, stage, data_loader, epoch):

        if stage == TRAIN_STAGE:
            device = self.device
            optimizer = self.optimizer
            MSE_criterion = self.criterion
            BCE_criterion = self.BCE_criterion
            epoch_loss = 0.0
            epoch_position_loss = 0.0
            epoch_domain_loss = 0.0
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

                positions, domain_preds = self.model(x)
                label_preds = positions[:source_x.shape[0]]

                domain_loss = BCE_criterion(domain_preds.squeeze(), domain_y)
                position_loss = MSE_criterion(label_preds, label_y)
                loss = position_loss - domain_loss*self.weight

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_position_loss += position_loss.item()
                epoch_domain_loss += domain_loss.item()
                # Print the loss and accuracy for the current batch
                if stage == TRAIN_STAGE and i % 50 == 0:
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
            return self.STL_model_evaluate(stage, data_loader, epoch)


# model和 classifier分离，共同train test
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

    def model_evaluate(self, stage, data_loader, epoch):

        if stage == TRAIN_STAGE:
            device = self.device
            optimizer = self.optimizer
            MSE_criterion = self.criterion
            BCE_criterion = self.BCE_criterion
            epoch_loss = 0.0
            epoch_position_loss = 0.0
            epoch_domain_loss = 0.0
            source_loader = self.train_loader
            target_loader = self.test_loader
            c_target_loader = cycle(target_loader)
            batches = zip(source_loader,  cycle(target_loader))
            first_batch = next(batches)
            n_batches = len(source_loader)
            for i, ((source_x, source_labels, index), (target_x, trage_y, index2)) in tqdm(enumerate(batches), total=n_batches):
                # initialize the x
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
                domain_loss_copy = domain_loss.clone().detach()
                position_loss = MSE_criterion(label_preds, label_y)

                self.discriminator_optimizer.zero_grad()
                domain_loss.backward(retain_graph=True)
                self.discriminator_optimizer.step()

                optimizer.zero_grad()
                loss = position_loss-domain_loss_copy*self.weight
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
            return self.STL_model_evaluate(stage, data_loader, epoch)


# 使用pretrain 只训练GAN，训练新model
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

                if i == n_batches:
                    break
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
                target_preds = self.discriminator(
                    targe_shared_features).squeeze()

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

# 使用pretrain 训练GAN和position，训练pretrainmodel


class MTL_ADDA_Trainer_with_pre2(ModelTrainer):
    def __init__(self, model, Dataset, optimizer, discriminator, scheduler=None, batch_size=64, n_epoch=15, weight=1, Trainer_name='Trainer') -> None:
        self.discriminator = discriminator
        self.weight = weight
        super().__init__(model, Dataset, optimizer,
                         scheduler, batch_size, n_epoch, Trainer_name)

    def initialization(self):
        super().initialization()
        self.discriminator.to(self.device)
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

    def model_evaluate(self, stage, data_loader, epoch):
        if stage == TRAIN_STAGE:
            device = self.device
            optimizer = self.optimizer
            MSE_criterion = self.criterion
            BCE_criterion = self.BCE_criterion
            epoch_loss = 0.0
            epoch_position_loss = 0.0
            epoch_domain_loss = 0.0
            epoch_trage_loss = 0.0
            source_loader = self.train_loader
            target_loader = self.test_loader
            batches = zip(source_loader,  cycle(target_loader))
            n_batches = len(source_loader)
            predictions_accuracy = 0.0
            epoch_log_domain_loss = 0.0
            for i, ((source_x, source_labels, *index), (target_x, trage_y, *index2)) in tqdm(enumerate(batches), total=n_batches):
                optimizer.zero_grad()
                x = torch.cat([source_x, target_x])
                x = x.to(device)
                domain_y = torch.cat([torch.ones(source_x.shape[0]),
                                      torch.zeros(target_x.shape[0])])
                domain_y = domain_y.to(device)
                label_y = source_labels.to(device)
                trage_y = trage_y.to(device)

                positions, sf = self.model(x)
                domain_preds = self.discriminator(sf).squeeze()
                label_preds = positions[:source_x.shape[0]]
                trage_loss = MSE_criterion(
                    positions[source_x.shape[0]:], trage_y)

                domain_loss = BCE_criterion(domain_preds, domain_y)
                # 计算准确率
                domain_preds_binary = (torch.sigmoid(
                    domain_preds) > 0.5).float()  # 将预测转换为二元标签
                correct = (domain_preds_binary ==
                           domain_y).float().sum()  # 计算正确预测的数量

                log_domain_loss = torch.log(domain_loss.clamp(min=1e-6))
                optimizer.zero_grad()
                position_loss = MSE_criterion(label_preds, label_y)
                loss = position_loss - log_domain_loss*self.weight
                loss.backward()
                optimizer.step()

                predictions_accuracy += correct.item()/len(domain_y)
                epoch_trage_loss += trage_loss.item()
                epoch_loss += loss.item()
                epoch_position_loss += position_loss.item()
                epoch_domain_loss += domain_loss.item()
                epoch_log_domain_loss += log_domain_loss.item()
                # Print the loss and accuracy for the current batch
                if stage == TRAIN_STAGE and i % 100 == 0:
                    print(f"\n Epoch {epoch}, Batch {i}\n overall_loss: { epoch_loss/(i+1)} \n" +
                          f" position loss: {epoch_position_loss/(i+1)} \n" +
                          f" Source RMSE(mm): {default_round(position_loss.item())} \n" +
                          f" domain loss: {domain_loss.item()}\n" +
                          f" epoch_log_domain_loss: {epoch_log_domain_loss/(i+1)} \n" +
                          f" domain acc: {predictions_accuracy / (i+1)}\n" +
                          f" trage RMSE: {Cal_RMSE(trage_loss.item())}\n")

            loss = {'overall_loss': epoch_loss / len(data_loader),
                    'position_loss': epoch_position_loss / len(data_loader),
                    'position_RMSE': Cal_RMSE(epoch_position_loss / len(data_loader)),
                    'domain_loss': domain_loss / len(data_loader)
                    }

            return loss

        # Test and Val stage is the same as Single Task Learning
        else:
            return self.STL_model_evaluate(stage, data_loader, epoch)

# 分离discriminatoer的训练
class MTL_ADDA_Trainer_with_pre_seper(ModelTrainer):
    def __init__(self, model, Dataset, optimizer, discriminator, scheduler=None, batch_size=64, n_epoch=15, weight=1, Trainer_name='Trainer') -> None:
        self.discriminator = discriminator
        self.weight = weight
        super().__init__(model, Dataset, optimizer,
                         scheduler, batch_size, n_epoch, Trainer_name)

    def initialization(self):
        super().initialization()
        self.discriminator.to(self.device)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=1e-4)
        self.discriminator_scheduler = torch.optim.lr_scheduler.StepLR(
            self.discriminator_optimizer, step_size=8, gamma=0.1)
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

    def train_discriminator(self, stage, data_loader, epoch):
        self.discriminator.train()
        self.model.eval()
        device = self.device
        BCE_criterion = self.BCE_criterion
        epoch_domain_loss = 0.0
        source_loader = self.train_loader
        # 假设 source_loader 和 target_loader 是两个 DataLoader
        # combined_loader = DataLoader(ConcatDataset([self.test_loader.dataset, self.val_loader.dataset]), batch_size=self.batch_size)
        batches = zip(source_loader,  cycle(self.test_loader.dataset))

        n_batches = len(source_loader)
        print(f"Get shared features of source and traget domain for training discriminator")
        # Generate shared features for the source and target domains
        shear_features = torch.tensor([]).to(device)
        domain_ys = torch.tensor([]).to(device)
        for i, ((source_x, source_labels, index), (target_x, trage_y, index2)) in tqdm(enumerate(batches), total=n_batches):
            x = torch.cat([source_x, target_x])
            x = x.to(device)
            domain_y = torch.cat([torch.ones(source_x.shape[0]),
                                  torch.zeros(target_x.shape[0])])
            domain_y = domain_y.to(device)
            domain_ys = torch.cat([domain_ys, domain_y])
            with torch.no_grad():
                positions, shear_feature = self.model(x)
            shear_features = torch.cat([shear_features, shear_feature])
        save_path = f'logs/{self.Trainer_name}/shared_features_epoch{epoch}.html'
        plot_shear_feature(shear_features, domain_ys, save_path)
        # train discriminator for num_epochs
        num_epochs = 48
        overall_loss = 0.0
        overall_acc = 0.0
        print(f"Train discriminator for {num_epochs} epochs")
        for i in tqdm(range(num_epochs), total=num_epochs):
            batch_size = self.batch_size
            epoch_predictions_accuracy = 0.0
            epoch_domain_loss = 0.0
            for sf_start in range(0, len(shear_features), batch_size):
                sf_end = sf_start + batch_size
                sf = shear_features[sf_start:sf_end]
                domain_y = domain_ys[sf_start:sf_end]
                domain_preds = self.discriminator(sf).squeeze()

                domain_loss = BCE_criterion(domain_preds, domain_y)
                self.discriminator_optimizer.zero_grad()
                domain_loss.backward()
                self.discriminator_optimizer.step()
                epoch_domain_loss += domain_loss.item()

                # 计算准确率
                domain_preds_binary = (torch.sigmoid(
                    domain_preds) > 0.5).float()  # 将预测转换为二元标签
                correct = (domain_preds_binary ==
                           domain_y).float().sum()  # 计算正确预测的数量
                epoch_predictions_accuracy += correct.item()/len(domain_y)

            overall_acc += epoch_predictions_accuracy / \
                (len(shear_features)/batch_size)
            overall_loss += epoch_domain_loss / \
                (len(shear_features)/batch_size)
        overall_loss /= num_epochs
        overall_acc /= num_epochs
        # Print the loss and accuracy for the current batch
        print(f"\n Epoch {epoch} \n tune_domain loss: {overall_loss}\n" +
              f" tune_domain acc: {overall_acc}\n")
        return {'tune_domain_loss': overall_loss,
                'tune_domain_acc': overall_acc}

    def train_model(self, stage, data_loader, epoch):

        device = self.device
        optimizer = self.optimizer
        MSE_criterion = self.criterion
        BCE_criterion = self.BCE_criterion
        epoch_loss = 0.0
        epoch_position_loss = 0.0
        epoch_domain_loss = 0.0
        epoch_log_domain_loss = 0.0
        source_loader = self.train_loader
        target_loader = self.test_loader

        dis_loss = self.train_discriminator(stage, data_loader, epoch)
        print(f"Training model with fix discriminator for batches")
        batches = zip(source_loader,  cycle(target_loader))
        predictions_accuracy = 0.0
        n_batches = len(source_loader)
        self.discriminator.eval()
        self.model.train()
        for i, ((source_x, source_labels, index), (target_x, trage_y, index2)) in tqdm(enumerate(batches), total=n_batches):
            optimizer.zero_grad()
            x = torch.cat([source_x, target_x])
            x = x.to(device)
            domain_y = torch.cat([torch.ones(source_x.shape[0]),
                                  torch.zeros(target_x.shape[0])])
            domain_y = domain_y.to(device)
            label_y = source_labels.to(device)
            positions, sf = self.model(x)
            with torch.no_grad():
                domain_preds = self.discriminator(sf).squeeze()
            label_preds = positions[:source_x.shape[0]]
            domain_loss = BCE_criterion(domain_preds, domain_y)

            optimizer.zero_grad()
            position_loss = MSE_criterion(label_preds, label_y)
            log_domain_loss = torch.log(domain_loss.clamp(min=1e-6))
            loss = position_loss - log_domain_loss*self.weight
            loss.backward()
            optimizer.step()
            # 计算准确率
            domain_preds_binary = (torch.sigmoid(
                domain_preds) > 0.5).float()  # 将预测转换为二元标签
            correct = (domain_preds_binary ==
                       domain_y).float().sum()  # 计算正确预测的数量
            predictions_accuracy += correct.item()/len(domain_y)
            epoch_log_domain_loss += log_domain_loss.item()

            epoch_loss += loss.item()
            epoch_position_loss += position_loss.item()
            epoch_domain_loss += domain_loss.item()

            # Print the loss and accuracy for the current batch
            if stage == TRAIN_STAGE and i % 100 == 0:
                print(f"\n Epoch {epoch}, Batch {i}\n overall_loss: { epoch_loss/(i+1)} \n" +
                      f" position loss: {epoch_position_loss/(i+1)} \n" +
                      f" Source RMSE(mm): { Cal_RMSE(epoch_position_loss / (i+1))} \n" +
                      f" domain loss: {domain_loss.item()}\n" +
                      f" epoch_log_domain_loss: {epoch_log_domain_loss/(i+1)} \n" +
                      f" domain acc: {predictions_accuracy / (i+1)}\n"
                      )

        loss = {'overall_loss': epoch_loss / len(data_loader),
                'position_loss': epoch_position_loss / len(data_loader),
                'position_RMSE': Cal_RMSE(epoch_position_loss / len(data_loader)),
                'domain_loss': epoch_domain_loss / len(data_loader),
                'log_domain_loss': epoch_log_domain_loss / len(data_loader),
                "domain_acc": predictions_accuracy / len(data_loader),
                }
        loss.update(dis_loss)
        return loss

    def model_evaluate(self, stage, data_loader, epoch):
        if stage == TRAIN_STAGE:
            return self.train_model(stage, data_loader, epoch)

        # Test and Val stage is the same as Single Task Learning
        else:
            return self.STL_model_evaluate(stage, data_loader, epoch)


# 在position shang adopt domain adversarial training
class MTL_position_ADDA(ModelTrainer):
    def __init__(self, model, Dataset, optimizer, discriminator, scheduler=None, batch_size=64, n_epoch=15, weight=1, Trainer_name='Trainer') -> None:
        self.discriminator = discriminator
        self.weight = weight
        super().__init__(model, Dataset, optimizer,
                         scheduler, batch_size, n_epoch, Trainer_name)

    def initialization(self):
        super().initialization()
        self.discriminator.to(self.device)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=1e-4)
        self.discriminator_scheduler = torch.optim.lr_scheduler.StepLR(
            self.discriminator_optimizer, step_size=8, gamma=0.1)
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

    def train_discriminator(self, stage, data_loader, epoch):
        self.discriminator.train()
        self.model.eval()
        device = self.device
        BCE_criterion = self.BCE_criterion
        epoch_domain_loss = 0.0
        source_loader = self.train_loader
        # 假设 source_loader 和 target_loader 是两个 DataLoader
        # combined_loader = DataLoader(ConcatDataset([self.test_loader.dataset, self.val_loader.dataset]), batch_size=self.batch_size)
        batches = zip(source_loader,  cycle(self.test_loader))

        n_batches = len(source_loader)
        print(f"Get shared features of source and traget domain for training discriminator")
        # Generate shared features for the source and target domains
        position_predicts = torch.tensor([]).to(device)
        domain_ys = torch.tensor([]).to(device)
        epoch_source_lables = torch.tensor([]).to(device)
        for i, ((source_x, source_labels, index), (target_x, trage_y, index2)) in tqdm(enumerate(batches), total=n_batches):
            x = torch.cat([source_x, target_x])
            x = x.to(device)
            domain_y = torch.cat([torch.ones(source_x.shape[0]),
                                  torch.zeros(target_x.shape[0])])
            domain_y = domain_y.to(device)
            domain_ys = torch.cat([domain_ys, domain_y])
            with torch.no_grad():
                positions, sf = self.model(x)
            position_predicts = torch.cat([position_predicts, positions])
            epoch_source_lables = torch.cat(
                [epoch_source_lables, source_labels])
        save_path = f'logs/{self.Trainer_name}/predict_position{epoch}.html'
        plot_positions(position_predicts, epoch_source_lables,
                       domain_ys, save_path)
        # train discriminator for num_epochs
        num_epochs = 48
        overall_loss = 0.0
        overall_acc = 0.0
        print(f"Train discriminator for {num_epochs} epochs")
        for i in tqdm(range(num_epochs), total=num_epochs):
            batch_size = self.batch_size
            epoch_predictions_accuracy = 0.0
            epoch_domain_loss = 0.0
            for sf_start in range(0, len(position_predicts), batch_size):
                sf_end = sf_start + batch_size
                sf = position_predicts[sf_start:sf_end]
                domain_y = domain_ys[sf_start:sf_end]
                domain_preds = self.discriminator(sf).squeeze()

                domain_loss = BCE_criterion(domain_preds, domain_y)
                self.discriminator_optimizer.zero_grad()
                domain_loss.backward()
                self.discriminator_optimizer.step()
                epoch_domain_loss += domain_loss.item()

                # 计算准确率
                domain_preds_binary = (torch.sigmoid(
                    domain_preds) > 0.5).float()  # 将预测转换为二元标签
                correct = (domain_preds_binary ==
                           domain_y).float().sum()  # 计算正确预测的数量
                epoch_predictions_accuracy += correct.item()/len(domain_y)

            overall_acc += epoch_predictions_accuracy / \
                (len(position_predicts)/batch_size)
            overall_loss += epoch_domain_loss / \
                (len(position_predicts)/batch_size)
        overall_loss /= num_epochs
        overall_acc /= num_epochs
        # Print the loss and accuracy for the current batch
        print(f"\n Epoch {epoch} \n tune_domain loss: {overall_loss}\n" +
              f" tune_domain acc: {overall_acc}\n")
        return {'tune_domain_loss': overall_loss,
                'tune_domain_acc': overall_acc}

    def train_model(self, stage, data_loader, epoch):

        device = self.device
        optimizer = self.optimizer
        MSE_criterion = self.criterion
        BCE_criterion = self.BCE_criterion
        epoch_loss = 0.0
        epoch_position_loss = 0.0
        epoch_domain_loss = 0.0
        epoch_log_domain_loss = 0.0
        source_loader = self.train_loader
        target_loader = self.test_loader

        dis_loss = self.train_discriminator(stage, data_loader, epoch)
        print(f"Training model with fix discriminator for batches")
        batches = zip(source_loader,  cycle(target_loader))
        predictions_accuracy = 0.0
        n_batches = len(source_loader)
        self.discriminator.eval()
        self.model.train()
        for i, ((source_x, source_labels, index), (target_x, trage_y, index2)) in tqdm(enumerate(batches), total=n_batches):
            optimizer.zero_grad()
            x = torch.cat([source_x, target_x])
            x = x.to(device)
            domain_y = torch.cat([torch.ones(source_x.shape[0]),
                                  torch.zeros(target_x.shape[0])])
            domain_y = domain_y.to(device)
            label_y = source_labels.to(device)
            positions, sf = self.model(x)
            with torch.no_grad():
                domain_preds = self.discriminator(positions).squeeze()
            label_preds = positions[:source_x.shape[0]]
            domain_loss = BCE_criterion(domain_preds, domain_y)

            optimizer.zero_grad()
            position_loss = MSE_criterion(label_preds, label_y)
            log_domain_loss = torch.log(domain_loss.clamp(min=1e-6))
            loss = position_loss - log_domain_loss*self.weight
            loss.backward()
            optimizer.step()
            # 计算准确率
            domain_preds_binary = (torch.sigmoid(
                domain_preds) > 0.5).float()  # 将预测转换为二元标签
            correct = (domain_preds_binary ==
                       domain_y).float().sum()  # 计算正确预测的数量
            predictions_accuracy += correct.item()/len(domain_y)
            epoch_log_domain_loss += log_domain_loss.item()

            epoch_loss += loss.item()
            epoch_position_loss += position_loss.item()
            epoch_domain_loss += domain_loss.item()

            # Print the loss and accuracy for the current batch
            if stage == TRAIN_STAGE and i % 100 == 0:
                print(f"\n Epoch {epoch}, Batch {i}\n overall_loss: { epoch_loss/(i+1)} \n" +
                      f" position loss: {epoch_position_loss/(i+1)} \n" +
                      f" Source RMSE(mm): { Cal_RMSE(epoch_position_loss / (i+1))} \n" +
                      f" domain loss: {domain_loss.item()}\n" +
                      f" epoch_log_domain_loss: {epoch_log_domain_loss/(i+1)} \n" +
                      f" domain acc: {predictions_accuracy / (i+1)}\n"
                      )

        loss = {'overall_loss': epoch_loss / len(data_loader),
                'position_loss': epoch_position_loss / len(data_loader),
                'position_RMSE': Cal_RMSE(epoch_position_loss / len(data_loader)),
                'domain_loss': epoch_domain_loss / len(data_loader),
                'log_domain_loss': epoch_log_domain_loss / len(data_loader),
                "domain_acc": predictions_accuracy / len(data_loader),
                }
        loss.update(dis_loss)
        return loss

    def model_evaluate(self, stage, data_loader, epoch):
        if stage == TRAIN_STAGE:
            return self.train_model(stage, data_loader, epoch)

        # Test and Val stage is the same as Single Task Learning
        else:
            return self.STL_model_evaluate(stage, data_loader, epoch)


# ADDA 分离loss，分离train test
class MTL_ADDA_Trainer5(ModelTrainer):
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

    def model_evaluate(self, stage, data_loader, epoch):

        if stage == TRAIN_STAGE:
            device = self.device
            optimizer = self.optimizer
            MSE_criterion = self.criterion
            BCE_criterion = self.BCE_criterion
            epoch_loss = 0.0
            epoch_position_loss = 0.0
            epoch_domain_loss = 0.0
            source_loader = self.train_loader
            target_loader = self.test_loader
            c_target_loader = cycle(target_loader)
            batches = zip(source_loader,  cycle(target_loader))
            first_batch = next(batches)
            n_batches = len(source_loader)
            # train model and classifier on train data
            for i, (source_x, source_labels, index) in tqdm(enumerate(self.train_loader)):
                # initialize the x

                domain_y = torch.ones(source_x.shape[0])
                source_x = source_x.to(device)
                domain_y = domain_y.to(device)
                label_y = source_labels.to(device)

                label_preds, shared_features = self.model(source_x)
                domain_preds = self.discriminator(shared_features).squeeze()

                domain_loss = BCE_criterion(domain_preds, domain_y)
                position_loss = MSE_criterion(label_preds, label_y)
                domain_loss_copied = domain_loss.clone().detach()
                self.discriminator_optimizer.zero_grad()
                domain_loss.backward(retain_graph=True)
                self.discriminator_optimizer.step()

                optimizer.zero_grad()
                loss = position_loss-domain_loss_copied*self.weight
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
            self.STL_model_evaluate(stage, data_loader, epoch)



class MTL_position_pupil_ADDA(ModelTrainer):
    def __init__(self, model, Dataset, optimizer, discriminator, scheduler=None, batch_size=64, n_epoch=15, weight=1, Trainer_name='Trainer') -> None:
        self.discriminator = discriminator
        self.weight = weight
        super().__init__(model, Dataset, optimizer,
                         scheduler, batch_size, n_epoch, Trainer_name)

    def initialization(self):
        super().initialization()
        self.discriminator.to(self.device)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=1e-4)
        self.discriminator_scheduler = torch.optim.lr_scheduler.StepLR(
            self.discriminator_optimizer, step_size=8, gamma=0.1)
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
        for i, (inputs, targets, pupil_size, *index) in enumerator:
            # Move the inputs and targets to the GPU (if available)
            inputs = inputs.to(device)
            targets = targets.to(device)
            # Compute the outputs and loss for the current batch
            if stage == TRAIN_STAGE:
                self.optimizer.zero_grad()
            outputs, pupil_size, sf = self.model(inputs)

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

    def train_discriminator(self, stage, data_loader, epoch):
        self.discriminator.train()
        self.model.eval()
        device = self.device
        BCE_criterion = self.BCE_criterion
        epoch_domain_loss = 0.0
        source_loader = self.train_loader
        # 假设 source_loader 和 target_loader 是两个 DataLoader
        # combined_loader = DataLoader(ConcatDataset([self.test_loader.dataset, self.val_loader.dataset]), batch_size=self.batch_size)
        batches = zip(source_loader,  cycle(self.test_loader))

        n_batches = len(source_loader)
        print(f"Get shared features of source and traget domain for training discriminator")
        # Generate shared features for the source and target domains
        position_predicts = torch.tensor([]).to(device)
        domain_ys = torch.tensor([]).to(device)
        epoch_source_lables = torch.tensor([]).to(device)
        epoch_shear_features = torch.tensor([]).to(device)
        for i, ((source_x, source_labels, pupilsize, index), (target_x, trage_y, pupilsize, index2)) in tqdm(enumerate(batches), total=n_batches):
            x = torch.cat([source_x, target_x])
            x = x.to(device)
            source_labels = source_labels.to(device)
            domain_y = torch.cat([torch.ones(source_x.shape[0]),
                                  torch.zeros(target_x.shape[0])])
            domain_y = domain_y.to(device)
            domain_ys = torch.cat([domain_ys, domain_y])
            with torch.no_grad():
                positions, pupilsize, sf = self.model(x)
            position_predicts = torch.cat([position_predicts, positions])
            epoch_source_lables = torch.cat(
                [epoch_source_lables, source_labels])
            epoch_shear_features = torch.cat([epoch_shear_features, sf])
        save_path = f'logs/{self.Trainer_name}/predict_position{epoch}.html'
        plot_positions_from_input(
            position_predicts, epoch_source_lables, domain_ys, save_path)
        save_path = f'logs/{self.Trainer_name}/shear_feature{epoch}.html'
        plot_shear_feature(epoch_shear_features, domain_ys, save_path)
        # train discriminator for num_epochs
        num_epochs = 48
        overall_loss = 0.0
        overall_acc = 0.0
        print(f"Train discriminator for {num_epochs} epochs")
        for i in tqdm(range(num_epochs), total=num_epochs):
            batch_size = self.batch_size
            epoch_predictions_accuracy = 0.0
            epoch_domain_loss = 0.0
            for sf_start in range(0, len(position_predicts), batch_size):
                sf_end = sf_start + batch_size
                batch_position = position_predicts[sf_start:sf_end]
                domain_y = domain_ys[sf_start:sf_end]
                outputs = self.discriminator(batch_position)
                domain_preds = outputs.squeeze()
                domain_loss = BCE_criterion(domain_preds, domain_y)
                self.discriminator_optimizer.zero_grad()
                domain_loss.backward()
                self.discriminator_optimizer.step()
                epoch_domain_loss += domain_loss.item()

                # 计算准确率
                domain_preds_binary = (torch.sigmoid(
                    domain_preds) > 0.5).float()  # 将预测转换为二元标签
                correct = (domain_preds_binary ==
                           domain_y).float().sum()  # 计算正确预测的数量
                epoch_predictions_accuracy += correct.item()/len(domain_y)

            overall_acc += epoch_predictions_accuracy / \
                (len(position_predicts)/batch_size)
            overall_loss += epoch_domain_loss / \
                (len(position_predicts)/batch_size)
        overall_loss /= num_epochs
        overall_acc /= num_epochs
        # Print the loss and accuracy for the current batch
        print(f"\n Epoch {epoch} \n tune_domain loss: {overall_loss}\n" +
              f" tune_domain acc: {overall_acc}\n")
        return {'tune_domain_loss': overall_loss,
                'tune_domain_acc': overall_acc}

    def train_model(self, stage, data_loader, epoch):

        device = self.device
        optimizer = self.optimizer
        MSE_criterion = self.criterion
        BCE_criterion = self.BCE_criterion
        epoch_loss = 0.0
        epoch_position_loss = 0.0
        epoch_domain_loss = 0.0
        epoch_log_domain_loss = 0.0
        epoch_pupil_loss = 0.0
        source_loader = self.train_loader
        target_loader = self.test_loader

        dis_loss = self.train_discriminator(stage, data_loader, epoch)
        print(f"Training model with fix discriminator for batches")
        batches = zip(source_loader,  cycle(target_loader))
        predictions_accuracy = 0.0
        n_batches = len(source_loader)
        self.discriminator.eval()
        self.model.train()
        for i, ((source_x, source_labels, pupil_size, index), (target_x, trage_y, pupil_size2, index2)) in tqdm(enumerate(batches), total=n_batches):
            optimizer.zero_grad()
            x = torch.cat([source_x, target_x])
            x = x.to(device)
            pupil_size = pupil_size.to(device)
            domain_y = torch.cat([torch.ones(source_x.shape[0]),
                                  torch.zeros(target_x.shape[0])])
            domain_y = domain_y.to(device)
            label_y = source_labels.to(device)
            positions, pred_pupil_size, sf = self.model(x)
            with torch.no_grad():
                domain_preds = self.discriminator(positions).squeeze()
            label_preds = positions[:source_x.shape[0]]
            pred_pupil_size = pred_pupil_size[:source_x.shape[0]]
            domain_loss = BCE_criterion(
                domain_preds[source_x.shape[0]:], domain_y[source_x.shape[0]:])

            optimizer.zero_grad()
            position_loss = MSE_criterion(label_preds, label_y)
            pupil_size_loss = MSE_criterion(pred_pupil_size, pupil_size)
            log_domain_loss = torch.log(domain_loss.clamp(min=1e-6))
            loss = position_loss - log_domain_loss*self.weight + pupil_size_loss*1000
            loss.backward()
            optimizer.step()
            # 计算准确率
            domain_preds_binary = (torch.sigmoid(
                domain_preds) > 0.5).float()  # 将预测转换为二元标签
            correct = (domain_preds_binary ==
                       domain_y).float().sum()  # 计算正确预测的数量
            predictions_accuracy += correct.item()/len(domain_y)
            epoch_log_domain_loss += log_domain_loss.item()

            epoch_loss += loss.item()
            epoch_position_loss += position_loss.item()
            epoch_domain_loss += domain_loss.item()
            epoch_pupil_loss += pupil_size_loss.item()
            # Print the loss and accuracy for the current batch
            if stage == TRAIN_STAGE and i % 100 == 0:
                print(f"\n Epoch {epoch}, Batch {i}\n overall_loss: { epoch_loss/(i+1)} \n" +
                      f" position loss: {epoch_position_loss/(i+1)} \n" +
                      f" Source RMSE(mm): { Cal_RMSE(epoch_position_loss / (i+1))} \n" +
                      f" domain loss: {domain_loss.item()}\n" +
                      f" epoch_log_domain_loss: {epoch_log_domain_loss/(i+1)} \n" +
                      f" domain acc: {predictions_accuracy / (i+1)}\n" +
                      f" pupil_size_loss: {epoch_pupil_loss/(i+1)}\n"
                      )

        loss = {'overall_loss': epoch_loss / len(data_loader),
                'position_loss': epoch_position_loss / len(data_loader),
                'position_RMSE': Cal_RMSE(epoch_position_loss / len(data_loader)),
                'domain_loss': epoch_domain_loss / len(data_loader),
                'log_domain_loss': epoch_log_domain_loss / len(data_loader),
                "domain_acc": predictions_accuracy / len(data_loader),
                "pupil_size_loss": epoch_pupil_loss / len(data_loader),
                }
        loss.update(dis_loss)
        return loss

    def model_evaluate(self, stage, data_loader, epoch):
        if stage == TRAIN_STAGE:
            return self.train_model(stage, data_loader, epoch)

        # Test and Val stage is the same as Single Task Learning
        else:
            return self.STL_model_evaluate(stage, data_loader, epoch)


class MTL_position_pupil_ADDA_v2(ModelTrainer):
    def __init__(self, model, Dataset, optimizer, discriminator, scheduler=None, batch_size=64, n_epoch=15, weight=1, Trainer_name='Trainer') -> None:
        self.discriminator = discriminator
        self.weight = weight
        super().__init__(model, Dataset, optimizer,
                         scheduler, batch_size, n_epoch, Trainer_name)

    def initialization(self):
        super().initialization()
        self.discriminator.to(self.device)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=1e-4)
        self.discriminator_scheduler = torch.optim.lr_scheduler.StepLR(
            self.discriminator_optimizer, step_size=8, gamma=0.1)
        self.BCE_criterion = nn.BCEWithLogitsLoss()
        self.BCE_criterion.to(self.device)
        self.pretrain_model = deepcopy(self.model)
        self.pretrain_model.to(self.device)
        # 计算复制 target_loader 的倍数，以使其长度不小于 source_loader


    def train_discriminator(self, stage, data_loader, epoch):
        self.discriminator.train()
        self.model.eval()
        device = self.device
        BCE_criterion = self.BCE_criterion
        epoch_domain_loss = 0.0
        source_loader = self.train_loader
        # 假设 source_loader 和 target_loader 是两个 DataLoader
        # combined_loader = DataLoader(ConcatDataset([self.test_loader.dataset, self.val_loader.dataset]), batch_size=self.batch_size)
        batches = zip(source_loader,  cycle(self.test_loader))

        n_batches = len(source_loader)
        print(f"Get shared features of source and traget domain for training discriminator")
        # Generate shared features for the source and target domains
        position_predicts = torch.tensor([]).to(device)
        domain_ys = torch.tensor([]).to(device)
        epoch_source_lables = torch.tensor([]).to(device)
        epoch_shear_features = torch.tensor([]).to(device)
        for i, ((source_x, source_labels, pupilsize, index), (target_x, trage_y, pupilsize, index2)) in tqdm(enumerate(batches), total=n_batches):

            source_x = source_x.to(device)
            target_x = target_x.to(device)

            source_labels = source_labels.to(device)
            domain_y = torch.cat([torch.ones(source_x.shape[0]),
                                  torch.zeros(target_x.shape[0])])
            domain_y = domain_y.to(device)
            domain_ys = torch.cat([domain_ys, domain_y])
            with torch.no_grad():
                source_positions, pupilsize, sf1 = self.pretrain_model(
                    source_x)
                traget_positions, pupilsize, sf2 = self.model(target_x)
            position_predicts = torch.cat(
                [position_predicts, source_positions, traget_positions])
            epoch_source_lables = torch.cat(
                [epoch_source_lables, source_labels])
            epoch_shear_features = torch.cat([epoch_shear_features, sf1, sf2])
        save_path = f'logs/{self.Trainer_name}/predict_position{epoch}.html'
        plot_positions_from_input(
            position_predicts, epoch_source_lables, domain_ys, save_path)
        save_path = f'logs/{self.Trainer_name}/shear_feature{epoch}.html'
        plot_shear_feature(epoch_shear_features, domain_ys, save_path)
        # train discriminator for num_epochs
        num_epochs = 48
        overall_loss = 0.0
        overall_acc = 0.0
        print(f"Train discriminator for {num_epochs} epochs")
        for i in tqdm(range(num_epochs), total=num_epochs):
            batch_size = self.batch_size
            epoch_predictions_accuracy = 0.0
            epoch_domain_loss = 0.0
            for sf_start in range(0, len(position_predicts), batch_size):
                sf_end = sf_start + batch_size
                batch_position = position_predicts[sf_start:sf_end]
                domain_y = domain_ys[sf_start:sf_end]
                outputs = self.discriminator(batch_position)
                domain_preds = outputs.squeeze()
                domain_loss = BCE_criterion(domain_preds, domain_y)
                self.discriminator_optimizer.zero_grad()
                domain_loss.backward()
                self.discriminator_optimizer.step()
                epoch_domain_loss += domain_loss.item()

                # 计算准确率
                domain_preds_binary = (torch.sigmoid(
                    domain_preds) > 0.5).float()  # 将预测转换为二元标签
                correct = (domain_preds_binary ==
                           domain_y).float().sum()  # 计算正确预测的数量
                epoch_predictions_accuracy += correct.item()/len(domain_y)

            overall_acc += epoch_predictions_accuracy / \
                (len(position_predicts)/batch_size)
            overall_loss += epoch_domain_loss / \
                (len(position_predicts)/batch_size)
        overall_loss /= num_epochs
        overall_acc /= num_epochs
        # Print the loss and accuracy for the current batch
        print(f"\n Epoch {epoch} \n tune_domain loss: {overall_loss}\n" +
              f" tune_domain acc: {overall_acc}\n")
        return {'tune_domain_loss': overall_loss,
                'tune_domain_acc': overall_acc}

    def train_model(self, stage, data_loader, epoch):

        device = self.device
        optimizer = self.optimizer
        MSE_criterion = self.criterion
        BCE_criterion = self.BCE_criterion
        epoch_loss = 0.0
        epoch_position_loss = 0.0
        epoch_domain_loss = 0.0
        epoch_log_domain_loss = 0.0
        epoch_pupil_loss = 0.0
        source_loader = self.train_loader
        target_loader = self.test_loader

        dis_loss = self.train_discriminator(stage, data_loader, epoch)
        print(f"Training model with fix discriminator for batches")
        batches = zip(source_loader,  cycle(target_loader))
        predictions_accuracy = 0.0
        n_batches = len(source_loader)
        self.discriminator.eval()
        self.model.train()
        for i, ((source_x, source_labels, pupil_size, index), (target_x, trage_y, pupil_size2, index2)) in tqdm(enumerate(batches), total=n_batches):
            optimizer.zero_grad()
            x = torch.cat([source_x, target_x])
            x = x.to(device)
            pupil_size = pupil_size.to(device)
            domain_y = torch.cat([torch.ones(source_x.shape[0]),
                                  torch.zeros(target_x.shape[0])])
            domain_y = domain_y.to(device)
            label_y = source_labels.to(device)
            positions, pred_pupil_size, sf = self.model(x)
            with torch.no_grad():
                domain_preds = self.discriminator(positions).squeeze()
            label_preds = positions[:source_x.shape[0]]
            pred_pupil_size = pred_pupil_size[:source_x.shape[0]]
            domain_loss = BCE_criterion(
                domain_preds[source_x.shape[0]:], domain_y[source_x.shape[0]:])

            optimizer.zero_grad()
            position_loss = MSE_criterion(label_preds, label_y)
            pupil_size_loss = MSE_criterion(pred_pupil_size, pupil_size)
            log_domain_loss = torch.log(domain_loss.clamp(min=1e-6))
            loss = position_loss - log_domain_loss*self.weight + pupil_size_loss*1000
            loss.backward()
            optimizer.step()
            # 计算准确率
            domain_preds_binary = (torch.sigmoid(
                domain_preds) > 0.5).float()  # 将预测转换为二元标签
            correct = (domain_preds_binary ==
                       domain_y).float().sum()  # 计算正确预测的数量
            predictions_accuracy += correct.item()/len(domain_y)
            epoch_log_domain_loss += log_domain_loss.item()

            epoch_loss += loss.item()
            epoch_position_loss += position_loss.item()
            epoch_domain_loss += domain_loss.item()
            epoch_pupil_loss += pupil_size_loss.item()
            # Print the loss and accuracy for the current batch
            if stage == TRAIN_STAGE and i % 100 == 0:
                print(f"\n Epoch {epoch}, Batch {i}\n overall_loss: { epoch_loss/(i+1)} \n" +
                      f" position loss: {epoch_position_loss/(i+1)} \n" +
                      f" Source RMSE(mm): { Cal_RMSE(epoch_position_loss / (i+1))} \n" +
                      f" domain loss: {domain_loss.item()}\n" +
                      f" epoch_log_domain_loss: {epoch_log_domain_loss/(i+1)} \n" +
                      f" domain acc: {predictions_accuracy / (i+1)}\n" +
                      f" pupil_size_loss: {epoch_pupil_loss/(i+1)}\n"
                      )

        loss = {'overall_loss': epoch_loss / len(data_loader),
                'position_loss': epoch_position_loss / len(data_loader),
                'position_RMSE': Cal_RMSE(epoch_position_loss / len(data_loader)),
                'domain_loss': epoch_domain_loss / len(data_loader),
                'log_domain_loss': epoch_log_domain_loss / len(data_loader),
                "domain_acc": predictions_accuracy / len(data_loader),
                "pupil_size_loss": epoch_pupil_loss / len(data_loader),
                }
        loss.update(dis_loss)
        return loss

    def model_evaluate(self, stage, data_loader, epoch):
        if stage == TRAIN_STAGE:
            return self.train_model(stage, data_loader, epoch)

        # Test and Val stage is the same as Single Task Learning
        else:
            return self.STL_model_evaluate(stage, data_loader, epoch)

# 训练一个新的模型，而不是在原模型上训练


class MTL_position_pupil_ADDA_v3(ModelTrainer):
    def __init__(self, model, Dataset, optimizer, discriminator, scheduler=None, batch_size=64, n_epoch=15, weight=1, Trainer_name='Trainer') -> None:
        self.discriminator = discriminator
        self.weight = weight
        super().__init__(model, Dataset, optimizer,
                         scheduler, batch_size, n_epoch, Trainer_name)

    def initialization(self):
        super().initialization()
        self.discriminator.to(self.device)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=1e-4)
        self.discriminator_scheduler = torch.optim.lr_scheduler.StepLR(
            self.discriminator_optimizer, step_size=8, gamma=0.1)
        self.BCE_criterion = nn.BCEWithLogitsLoss()
        self.BCE_criterion.to(self.device)
        self.pretrain_model = deepcopy(self.model)
        self.pretrain_model.to(self.device)
        self.pretrain_model.eval()
        # 计算复制 target_loader 的倍数，以使其长度不小于 source_loader

    def STL_model_evaluate(self, stage, data_loader, epoch):
        device = self.device
        optimizer = self.optimizer
        epoch_loss = 0.0
        epoch_position_loss = 0.0

        enumerator = tqdm(enumerate(data_loader)
                          ) if stage == TRAIN_STAGE else enumerate(data_loader)
        criterion = self.criterion
        for i, (inputs, targets, pupil_size, *index) in enumerator:
            # Move the inputs and targets to the GPU (if available)
            inputs = inputs.to(device)
            targets = targets.to(device)
            # Compute the outputs and loss for the current batch
            if stage == TRAIN_STAGE:
                self.optimizer.zero_grad()
            outputs, pupil_size, sf = self.model(inputs)

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

    def train_discriminator(self, stage, data_loader, epoch):
        self.discriminator.train()
        self.model.eval()
        device = self.device
        BCE_criterion = self.BCE_criterion
        epoch_domain_loss = 0.0
        source_loader = self.train_loader
        # 假设 source_loader 和 target_loader 是两个 DataLoader
        # combined_loader = DataLoader(ConcatDataset([self.test_loader.dataset, self.val_loader.dataset]), batch_size=self.batch_size)
        batches = zip(source_loader,  cycle(self.test_loader))

        n_batches = len(source_loader)
        print(f"Get shared features of source and traget domain for training discriminator")
        # Generate shared features for the source and target domains
        position_predicts = torch.tensor([]).to(device)
        domain_ys = torch.tensor([]).to(device)
        epoch_source_lables = torch.tensor([]).to(device)
        epoch_shear_features = torch.tensor([]).to(device)
        for i, ((source_x, source_labels, pupilsize, index), (target_x, trage_y, pupilsize, index2)) in tqdm(enumerate(batches), total=n_batches):
            source_x = source_x.to(device)
            target_x = target_x.to(device)
            source_labels = source_labels.to(device)
            domain_y = torch.cat([torch.ones(source_x.shape[0]),
                                  torch.zeros(target_x.shape[0])])
            domain_y = domain_y.to(device)
            domain_ys = torch.cat([domain_ys, domain_y])
            with torch.no_grad():
                source_positions, pupilsize1, sf1 = self.pretrain_model(
                    source_x)
                target_positions, pupilsize2, sf2 = self.model(target_x)
            position_predicts = torch.cat(
                [position_predicts, source_positions, target_positions])
            epoch_source_lables = torch.cat(
                [epoch_source_lables, source_labels])
            epoch_shear_features = torch.cat([epoch_shear_features, sf1, sf2])
        save_path = f'logs/{self.Trainer_name}/predict_position{epoch}.html'
        plot_positions_from_input(
            position_predicts, epoch_source_lables, domain_ys, save_path)
        save_path = f'logs/{self.Trainer_name}/shear_feature{epoch}.html'
        plot_shear_feature(epoch_shear_features, domain_ys, save_path)
        # train discriminator for num_epochs
        num_epochs = 48
        overall_loss = 0.0
        overall_acc = 0.0
        print(f"Train discriminator for {num_epochs} epochs")
        for i in tqdm(range(num_epochs), total=num_epochs):
            batch_size = self.batch_size
            epoch_predictions_accuracy = 0.0
            epoch_domain_loss = 0.0
            for sf_start in range(0, len(position_predicts), batch_size):
                sf_end = sf_start + batch_size
                batch_position = position_predicts[sf_start:sf_end]
                domain_y = domain_ys[sf_start:sf_end]
                outputs = self.discriminator(batch_position)
                domain_preds = outputs.squeeze()
                domain_loss = BCE_criterion(domain_preds, domain_y)
                self.discriminator_optimizer.zero_grad()
                domain_loss.backward()
                self.discriminator_optimizer.step()
                epoch_domain_loss += domain_loss.item()

                # 计算准确率
                domain_preds_binary = (torch.sigmoid(
                    domain_preds) > 0.5).float()  # 将预测转换为二元标签
                correct = (domain_preds_binary ==
                           domain_y).float().sum()  # 计算正确预测的数量
                epoch_predictions_accuracy += correct.item()/len(domain_y)

            overall_acc += epoch_predictions_accuracy / \
                (len(position_predicts)/batch_size)
            overall_loss += epoch_domain_loss / \
                (len(position_predicts)/batch_size)
        overall_loss /= num_epochs
        overall_acc /= num_epochs
        # Print the loss and accuracy for the current batch
        print(f"\n Epoch {epoch} \n tune_domain loss: {overall_loss}\n" +
              f" tune_domain acc: {overall_acc}\n")
        return {'tune_domain_loss': overall_loss,
                'tune_domain_acc': overall_acc}

    def train_model(self, stage, data_loader, epoch):

        device = self.device
        optimizer = self.optimizer
        MSE_criterion = self.criterion
        BCE_criterion = self.BCE_criterion
        epoch_loss = 0.0
        epoch_position_loss = 0.0
        epoch_domain_loss = 0.0
        epoch_pupil_loss = 0.0
        source_loader = self.train_loader
        target_loader = self.test_loader

        dis_loss = self.train_discriminator(stage, data_loader, epoch)
        print(f"Training model with fix discriminator for batches")
        batches = zip(source_loader,  cycle(target_loader))
        predictions_accuracy = 0.0
        n_batches = len(source_loader)
        self.discriminator.eval()
        self.model.train()
        for i, ((source_x, source_labels, pupil_size, index), (target_x, trage_y, pupil_size2, index2)) in tqdm(enumerate(batches), total=n_batches):
            optimizer.zero_grad()
            x = torch.cat([source_x, target_x])
            x = x.to(device)
            pupil_size = pupil_size.to(device)
            domain_y = torch.cat(
                [torch.ones(source_x.shape[0]), torch.zeros(target_x.shape[0])])

            domain_y = domain_y.to(device)
            label_y = source_labels.to(device)
            positions, pred_pupil_size, sf = self.model(x)

            domain_preds = self.discriminator(positions).squeeze()

            label_preds = positions[:source_x.shape[0]]
            pred_pupil_size = pred_pupil_size[:source_x.shape[0]]
            domain_loss = BCE_criterion(domain_preds, domain_y)

            optimizer.zero_grad()
            position_loss = MSE_criterion(label_preds, label_y)
            pupil_size_loss = MSE_criterion(pred_pupil_size, pupil_size)
            loss = - domain_loss
            loss.backward()
            optimizer.step()
            # 计算准确率
            domain_preds_binary = (torch.sigmoid(
                domain_preds) > 0.5).float()  # 将预测转换为二元标签
            correct = (domain_preds_binary ==
                       domain_y).float().sum()  # 计算正确预测的数量
            predictions_accuracy += correct.item()/len(domain_y)

            epoch_loss += loss.item()
            epoch_position_loss += position_loss.item()
            epoch_domain_loss += domain_loss.item()
            epoch_pupil_loss += pupil_size_loss.item()
            # Print the loss and accuracy for the current batch
            if stage == TRAIN_STAGE and i % 100 == 0:
                print(f"\n Epoch {epoch}, Batch {i}\n overall_loss: { epoch_loss/(i+1)} \n" +
                      f" position loss: {epoch_position_loss/(i+1)} \n" +
                      f" Source RMSE(mm): { Cal_RMSE(epoch_position_loss / (i+1))} \n" +
                      f" domain loss: {domain_loss.item()}\n" +
                      f" domain acc: {predictions_accuracy / (i+1)}\n" +
                      f" pupil_size_loss: {epoch_pupil_loss/(i+1)}\n"
                      )

        loss = {'overall_loss': epoch_loss / len(data_loader),
                'position_loss': epoch_position_loss / len(data_loader),
                'position_RMSE': Cal_RMSE(epoch_position_loss / len(data_loader)),
                'domain_loss': epoch_domain_loss / len(data_loader),
                "domain_acc": predictions_accuracy / len(data_loader),
                "pupil_size_loss": epoch_pupil_loss / len(data_loader),
                }
        loss.update(dis_loss)
        return loss

    def model_evaluate(self, stage, data_loader, epoch):
        if stage == TRAIN_STAGE:
            return self.train_model(stage, data_loader, epoch)

        # Test and Val stage is the same as Single Task Learning
        else:
            return self.STL_model_evaluate(stage, data_loader, epoch)

# Use pointTransformer as discriminator
class MTL_position_pupil_ADDA_pointnet(ModelTrainer):
    def __init__(self, model, Dataset, optimizer, discriminator, scheduler=None, batch_size=64, n_epoch=15, weight=1, Trainer_name='Trainer') -> None:
        self.discriminator = discriminator
        self.weight = weight
        super().__init__(model, Dataset, optimizer,
                         scheduler, batch_size, n_epoch, Trainer_name)

    def initialization(self):
        super().initialization(drop_last=True)
        self.discriminator.to(self.device)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=1e-4)
        self.discriminator_scheduler = torch.optim.lr_scheduler.StepLR(
            self.discriminator_optimizer, step_size=8, gamma=0.1)
        self.BCE_criterion = nn.BCELoss()
        self.BCE_criterion.to(self.device)
        self.pretrain_model = deepcopy(self.model)
        self.pretrain_model.to(self.device)
        self.pretrain_model.eval()
        # 计算复制 target_loader 的倍数，以使其长度不小于 source_loader

    def STL_model_evaluate(self, stage, data_loader, epoch):
        device = self.device
        optimizer = self.optimizer
        epoch_loss = 0.0
        epoch_position_loss = 0.0

        enumerator = tqdm(enumerate(data_loader)
                          ) if stage == TRAIN_STAGE else enumerate(data_loader)
        criterion = self.criterion
        epoch_pred_position = torch.tensor([]).to(self.device)
        epoch_lables = torch.tensor([]).to(self.device)
        for i, (inputs, targets, pupil_size, *index) in enumerator:
            # Move the inputs and targets to the GPU (if available)
            inputs = inputs.to(device)
            targets = targets.to(device)
            # Compute the outputs and loss for the current batch
            if stage == TRAIN_STAGE:
                self.optimizer.zero_grad()
            outputs, pupil_size, sf = self.model(inputs)

            # loss = criterion(outputs.squeeze(), targets.squeeze())
            position_loss = criterion(outputs.squeeze(), targets.squeeze())
            # Compute the gradients and update the parameters
            if stage == TRAIN_STAGE:
                position_loss.backward()
                optimizer.step()
            epoch_loss += position_loss.item()
            epoch_position_loss += position_loss.item()
            epoch_pred_position = torch.cat([epoch_pred_position, outputs])
            epoch_lables = torch.cat([epoch_lables, targets])
            # Print the loss and accuracy for the current batch
            if stage == TRAIN_STAGE and i % 100 == 0:
                print(f"Epoch {epoch}, Batch {i}, position loss: {position_loss.item()}" +
                      f" RMSE(mm): {default_round(Cal_RMSE(position_loss.item()))}")

        loss = {'overall_loss': epoch_loss / len(data_loader),
                'position_loss': epoch_position_loss / len(data_loader),
                'position_RMSE': Cal_RMSE(epoch_position_loss / len(data_loader)),
                }
        if stage == VAL_STAGE:
            self.val_results = epoch_pred_position
            self.val_lables = epoch_lables
        if stage == TEST_STAGE:
            self.test_results = epoch_pred_position
            self.test_lables = epoch_lables
            save_path = f'logs/{self.Trainer_name}/predict_position{epoch}.html'
            plot_positions(self, save_path)

        if stage in [TEST_STAGE, VAL_STAGE]:
            print(
                f"Epoch {epoch}, {stage} Loss: {default_round(loss['overall_loss'])}, RMSE(mm): {default_round(loss['position_RMSE'])}")
        return loss

    def train_discriminator(self, stage, data_loader, epoch):
        self.discriminator.train()
        self.model.eval()
        device = self.device
        BCE_criterion = self.BCE_criterion
        epoch_domain_loss = 0.0
        source_loader = self.train_loader
        # 假设 source_loader 和 target_loader 是两个 DataLoader
        # combined_loader = DataLoader(ConcatDataset([self.test_loader.dataset, self.val_loader.dataset]), batch_size=self.batch_size)
        # TODO: 保证每个batch 都能有相同数目的点给discriminator
        batches = zip(source_loader,  cycle(self.test_loader))

        n_batches = len(source_loader)
        print(f"Get shared features of source and traget domain for training discriminator")
        # Generate shared features for the source and target domains
        position_predicts = torch.tensor([]).to(device)
        domain_ys = torch.tensor([]).to(device)
        epoch_source_lables = torch.tensor([]).to(device)
        epoch_shear_features = torch.tensor([]).to(device)

        for i, ((source_x, source_labels, pupilsize, index), (target_x, trage_y, pupilsize, index2)) in tqdm(enumerate(batches), total=n_batches):
            if i == n_batches:
                break
            source_x = source_x.to(device)
            target_x = target_x.to(device)
            source_labels = source_labels.to(device)
            domain_y = torch.cat([get01('source', 1, device),
                                  get01('Target', 1, device)])
            domain_y = domain_y.to(device)
            domain_ys = torch.cat([domain_ys, domain_y])
            with torch.no_grad():
                source_positions, pupilsize1, sf1 = self.pretrain_model(
                    source_x)
                target_positions, pupilsize2, sf2 = self.model(target_x)
            # 转到3d方便pointnet学习

            source_positions = torch.cat([source_positions, torch.ones(
                (source_positions.shape[0], 1)).to(device)], dim=1)
            target_positions = torch.cat([target_positions, torch.ones(
                (target_positions.shape[0], 1)).to(device)], dim=1)
            position_predicts = torch.cat(
                [position_predicts, source_positions.unsqueeze(0), target_positions.unsqueeze(0)])

        # train discriminator for num_epochs
        num_epochs = 48
        overall_loss = 0.0
        overall_acc = 0.0
        total_size, num_points, position_dim = position_predicts.shape
        feats = torch.ones(total_size, num_points,
                           self.discriminator.feature_dim).to(device)
        masks = torch.ones(total_size, num_points).bool().to(device)

        print(f"Train discriminator for {num_epochs} epochs")
        for i in tqdm(range(num_epochs), total=num_epochs):
            # discriminator_batch_size
            batch_size = 24
            epoch_predictions_accuracy = 0.0
            epoch_domain_loss = 0.0
            for sf_start in range(0, len(position_predicts), batch_size):
                sf_end = sf_start + batch_size
                batch_position = position_predicts[sf_start:sf_end]
                domain_y = domain_ys[sf_start:sf_end]
                feat = feats[sf_start:sf_end]
                mask = masks[sf_start:sf_end]
                outputs = self.discriminator(
                    feats=feat, x=batch_position, mask=mask)
                domain_preds = outputs
                # TODO: y shape也要变为1
                domain_loss = BCE_criterion(domain_preds.squeeze(1), domain_y)
                self.discriminator_optimizer.zero_grad()
                domain_loss.backward()
                self.discriminator_optimizer.step()
                epoch_domain_loss += domain_loss.item()

                # 计算准确率
                domain_preds_binary = (domain_preds.squeeze(
                    1) > 0.5).float()  # 将预测转换为二元标签
                correct = (domain_preds_binary ==
                           domain_y).float().sum()  # 计算正确预测的数量
                epoch_predictions_accuracy += correct.item()/len(domain_y)

            overall_acc += epoch_predictions_accuracy / \
                (len(position_predicts)/batch_size)
            overall_loss += epoch_domain_loss / \
                (len(position_predicts)/batch_size)
        overall_loss /= num_epochs
        overall_acc /= num_epochs
        # Print the loss and accuracy for the current batch
        print(f"\n Epoch {epoch} \n tune_domain loss: {overall_loss}\n" +
              f" tune_domain acc: {overall_acc}\n")
        return {'tune_domain_loss': overall_loss,
                'tune_domain_acc': overall_acc}

    def train_model(self, stage, data_loader, epoch):

        device = self.device
        optimizer = self.optimizer
        MSE_criterion = self.criterion
        BCE_criterion = self.BCE_criterion
        epoch_loss = 0.0
        epoch_position_loss = 0.0
        epoch_domain_loss = 0.0
        epoch_pupil_loss = 0.0
        epoch_log_domain_loss = 0.0
        source_loader = self.train_loader
        target_loader = self.test_loader

        dis_loss = self.train_discriminator(stage, data_loader, epoch)
        print(f"Training model with fix discriminator for batches")
        batches = zip(source_loader,  cycle(target_loader))
        predictions_accuracy = 0.0

        n_batches = len(source_loader)
        self.discriminator.eval()
        self.model.train()
        target_positions = torch.tensor([]).to(device)

        if stage == TRAIN_STAGE:
            self.train_results = torch.tensor([]).to(self.device)
            self.test_results = torch.tensor([]).to(self.device)
            self.val_results = torch.tensor([]).to(self.device)
            self.test_lables = torch.tensor([]).to(self.device)
            self.val_lables = torch.tensor([]).to(self.device)
            self.train_lables = torch.tensor([]).to(self.device)
        epoch_pred_position = torch.tensor([]).to(self.device)
        epoch_lables = torch.tensor([]).to(self.device)
        for i, ((source_x, source_labels, pupil_size, index), (target_x, trage_y, pupil_size2, index2)) in tqdm(enumerate(batches), total=n_batches):
            optimizer.zero_grad()
            source_x = source_x.to(device)
            target_x = target_x.to(device)
            pupil_size = pupil_size.to(device)

            source_labels = source_labels.to(device)
            label_y = source_labels.to(device)

            target_positions, target_pupil_size, sf = self.model(target_x)
            source_positions, source_pupil_size, sf = self.model(source_x)
            epoch_pred_position = torch.cat(
                [epoch_pred_position, source_positions])
            epoch_lables = torch.cat([epoch_lables, source_labels])
            source_positions_3d = torch.cat([source_positions,
                                             torch.ones((source_positions.shape[0], 1)).to(device)], dim=1).unsqueeze(0)
            target_positions_3d = torch.cat([target_positions,
                                             torch.ones((target_positions.shape[0], 1)).to(device)], dim=1).unsqueeze(0)

            source_domain_preds = self.discriminator(x=source_positions_3d,
                                                     feats=torch.ones(
                                                         1, self.batch_size, self.discriminator.feature_dim).to(device),
                                                     mask=torch.ones(1, self.batch_size).bool().to(device)).squeeze(1)
            target_domain_preds = self.discriminator(x=target_positions_3d,
                                                     feats=torch.ones(
                                                         1, self.batch_size, self.discriminator.feature_dim).to(device),
                                                     mask=torch.ones(1, self.batch_size).bool().to(device)).squeeze(1)

            target_domain_loss = BCE_criterion(
                target_domain_preds, get01('Target', 1, device))
            source_domian_loss = BCE_criterion(
                source_domain_preds, get01('source', 1, device))
            optimizer.zero_grad()
            position_loss = MSE_criterion(source_positions, label_y)
            pupil_size_loss = MSE_criterion(source_pupil_size, pupil_size)
            log_domain_loss = torch.log(target_domain_loss.clamp(min=1e-6))
            source_domian_loss = torch.log(source_domian_loss.clamp(min=1e-6))
            loss = position_loss + \
                (source_domian_loss-log_domain_loss) * \
                self.weight + pupil_size_loss*1000
            loss.backward()
            optimizer.step()
            # 计算准确率
            domain_preds_binary = (torch.sigmoid(
                source_domain_preds) > 0.5).float()  # 将预测转换为二元标签
            correct = (domain_preds_binary ==
                       source_domain_preds).float().sum()  # 计算正确预测的数量
            predictions_accuracy += correct.item()

            domain_loss = target_domain_loss
            epoch_loss += loss.item()
            epoch_position_loss += position_loss.item()
            epoch_domain_loss += domain_loss.item()
            epoch_pupil_loss += pupil_size_loss.item()
            epoch_log_domain_loss += source_domian_loss.item()-log_domain_loss.item()
            # Print the loss and accuracy for the current batch
            if stage == TRAIN_STAGE and i % 100 == 0:
                print(f"\n Epoch {epoch}, Batch {i}\n overall_loss: { epoch_loss/(i+1)} \n" +
                      f" position loss: {epoch_position_loss/(i+1)} \n" +
                      f" Source RMSE(mm): { Cal_RMSE(epoch_position_loss / (i+1))} \n" +
                      f" domain loss: {domain_loss.item()}\n" +
                      f" epoch_log_domain_loss: {epoch_log_domain_loss/(i+1)} \n" +
                      f" domain acc: {predictions_accuracy / (i+1)}\n" +
                      f" pupil_size_loss: {epoch_pupil_loss/(i+1)}\n"
                      )
        if stage == TRAIN_STAGE:
            self.train_results = epoch_pred_position
            self.train_lables = epoch_lables
        loss = {'overall_loss': epoch_loss / len(data_loader),
                'position_loss': epoch_position_loss / len(data_loader),
                'position_RMSE': Cal_RMSE(epoch_position_loss / len(data_loader)),
                'domain_loss': epoch_domain_loss / len(data_loader),
                'log_domain_loss': epoch_log_domain_loss / len(data_loader),
                "domain_acc": predictions_accuracy / len(data_loader),
                "pupil_size_loss": epoch_pupil_loss / len(data_loader),
                }
        loss.update(dis_loss)
        return loss

    def model_evaluate(self, stage, data_loader, epoch):
        if stage == TRAIN_STAGE:
            return self.train_model(stage, data_loader, epoch)

        # Test and Val stage is the same as Single Task Learning
        else:
            return self.STL_model_evaluate(stage, data_loader, epoch)




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


def get01(domian, shape, device):
    if domian == 'Target':
        return torch.zeros(shape).to(device)
    else:
        return torch.ones(shape).to(device)
# endregion
