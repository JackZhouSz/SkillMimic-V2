from isaacgym.torch_utils import *
from skillmimic.utils import torch_utils

import os
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader, random_split
import argparse


def compute_humanoid_observations(root_pos, root_rot, body_pos):
    root_h_obs = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    
    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:] # remove root pos

    obs = torch.cat((root_h_obs, local_body_pos), dim=-1)
    return obs

def compute_obj_observations(root_pos, root_rot, tar_pos):
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    
    local_tar_pos = tar_pos - root_pos
    local_tar_pos[..., -1] = tar_pos[..., -1]
    local_tar_pos = quat_rotate(heading_rot, local_tar_pos)

    return local_tar_pos

class HistoryEncoder(nn.Module):
    # def __init__(self, history_length):
    #     super(HistoryEncoder, self).__init__()
    #     self.conv1 = nn.Conv1d(in_channels=316, out_channels=128, kernel_size=3, stride=1, padding=1)
    #     self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
    #     self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
    #     self.flatten = nn.Flatten()
    #     self.fc = nn.Linear(32 * history_length, 3)
    def __init__(self, history_length, input_size, embedding_dim):
        super(HistoryEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * history_length, embedding_dim)

        # self.dropout = nn.Dropout(p=0.1) #Zd

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to (batch, channels, sequence_length)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.flatten(x)

        x = self.fc(x)

        # self.history_features = self.fc(x)
        # x = self.dropout(self.history_features) #Zd
        return x

class ComprehensiveModel(pl.LightningModule):
    # def __init__(self, history_length):
    #     super(ComprehensiveModel, self).__init__()
    #     self.history_encoder = HistoryEncoder(history_length)
    #     self.fc1 = nn.Linear(3 + 316 + 64, 512)
    #     self.fc2 = nn.Linear(512, 256)
    #     self.fc3 = nn.Linear(256, 316)
    def __init__(self, history_length, input_size=316, embedding_dim=3, lr=0.001):
        super(ComprehensiveModel, self).__init__()
        self.save_hyperparameters()
        self.history_encoder = HistoryEncoder(history_length, input_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim + input_size + 64, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 316)

    def forward(self, history, current_motion, current_label):
        history_features = self.history_encoder(history)
        # history_features = torch.zeros_like(history_features)
        x = torch.cat((history_features, current_motion, current_label), dim=1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        history, current_motion, current_label, y = batch
        y_hat = self(history, current_motion, current_label)
        # loss = nn.functional.mse_loss(y_hat, y) #+ torch.sum(self.history_encoder.history_features**2) * 1e-3 #Zd
        # self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        # return loss
    
        # Separate humanoid and obj parts (last 3 dimensions are obj)
        y_hat_humanoid, y_hat_obj = y_hat[..., :-3], y_hat[..., -3:]
        y_humanoid, y_obj = y[..., :-3], y[..., -3:]
        # Convert one-hot label back to skill_number
        skill_number = torch.argmax(current_label, dim=1)
        # Only calculate obj loss for samples that are not 0 or 10
        mask = ~((skill_number == 0) | (skill_number == 10))
        loss_humanoid = nn.functional.mse_loss(y_hat_humanoid, y_humanoid)
        if mask.any():
            loss_obj = nn.functional.mse_loss(y_hat_obj[mask], y_obj[mask])
        else:
            loss_obj = 0.0
        loss = loss_humanoid + loss_obj
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        history, current_motion, current_label, y = batch
        y_hat = self(history, current_motion, current_label)
        # loss = nn.functional.mse_loss(y_hat, y) #+ torch.sum(self.history_encoder.history_features**2) * 1e-3 #Zd
        # self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        # return loss

        # Separate humanoid and obj parts (last 3 dimensions are obj)
        y_hat_humanoid, y_hat_obj = y_hat[..., :-3], y_hat[..., -3:]
        y_humanoid, y_obj = y[..., :-3], y[..., -3:]
        skill_number = torch.argmax(current_label, dim=1)
        mask = ~((skill_number == 0) | (skill_number == 10))
        loss_humanoid = nn.functional.mse_loss(y_hat_humanoid, y_humanoid)
        if mask.any():
            loss_obj = nn.functional.mse_loss(y_hat_obj[mask], y_obj[mask])
        else:
            loss_obj = 0.0
        loss = loss_humanoid + loss_obj
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5),
            'monitor': 'train_loss',  # Monitor validation loss for scheduling #Zv
        }
        return [optimizer], [scheduler]


class CustomDataset(Dataset):
    def __init__(self, motion_dir, history_length=30):
        self.history_length = history_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_dof = 52
        # self.file_paths = [os.path.join(motion_dir, f) for f in os.listdir(motion_dir) if 'pickle' in f and f.endswith('.pt')]
        self.file_paths = [motion_dir] if os.path.isfile(motion_dir) else [ \
            os.path.join(root, f) 
            for root, dirs, filenames in os.walk(motion_dir) 
            for f in filenames 
            if 'pickle' in f and f.endswith('.pt')
        ]
        self.data = []
        
        for file_path in self.file_paths:
            print(file_path)
            source_data = torch.load(file_path)  # (seq_len, 337)
            source_state = self.data_to_state(source_data) # (seq_len, 808)
            nframe, dim = source_state.shape

            current_motion_data = source_state[:-1]
            target_data = source_state[1:]
            
            skill_number = int(os.path.basename(file_path).split('_')[0].strip('pickle'))
            current_label_data = torch.nn.functional.one_hot(torch.tensor(skill_number), num_classes=64)
            
            history_data = torch.zeros(nframe, history_length, dim)
            for i in range(current_motion_data.shape[0]):
                if i < history_length:
                    history_data[i, history_length-i:] = source_state[:i]
                else:
                    history_data[i] = source_state[i-history_length:i]
                self.data.append((history_data[i], current_motion_data[i], current_label_data, target_data[i], skill_number))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        history, current_motion, current_label, target_data, skill_number = self.data[idx]

        # To avoid modifying the original tensor, clone first
        history = history.clone()
        current_motion = current_motion.clone()
        current_label = current_label.clone()
        target_data = target_data.clone()

        # If skill_number is 0 or 10, replace the last 3 dimensions (obj part) with random numbers
        # This example only randomizes the obj position in 3 dimensions
        # If obj takes more dimensions in the actual state, adjust accordingly
        if skill_number in [0, 10]:
            # Last 3 dimensions are obj_pos, range [-5,5]
            current_motion[-3:] = torch.rand(3, device=self.device) * 10 - 5
            # Target can be random too, or consistent with current_motion
            # Here we demonstrate different from current_motion
            target_data[-3:] = torch.rand(3, device=self.device) * 10 - 5

        return history, current_motion, current_label, target_data
    
    def data_to_state(self, data):
        nframes = data.shape[0]
        root_pos = data[:, :3]
        root_rot = data[:, 3:6]
        body_pos = data[:, 165:165+53*3].reshape(nframes, 53, 3)
        humanoid_obs = compute_humanoid_observations(root_pos, root_rot, body_pos) # (nframes, 157)
        humanoid_obs = torch.cat((humanoid_obs, data[:, 9:9+156]), dim=-1) # (nframes, 313)
        obj_pos = data[:, 324:327] # (nframes, 3)
        obj_obs = compute_obj_observations(root_pos, root_rot, obj_pos) # (nframes, 3)
        state = torch.cat((humanoid_obs, obj_obs), dim=-1) # (nframes, 316)
        return state

class MotionDataModule(pl.LightningDataModule):
    def __init__(self, folder_path, window_size, batch_size=32):
        super().__init__()
        self.folder_path = folder_path
        self.window_size = window_size
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Initialize dataset and split here
        dataset = CustomDataset(self.folder_path, self.window_size)
        
        # Randomly split into training and validation sets
        train_size = int(0.85 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

class MotionDataModuleAll4Train(pl.LightningDataModule):
    def __init__(self, folder_path, window_size, batch_size=32):
        super().__init__()
        self.folder_path = folder_path
        self.window_size = window_size
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Initialize dataset here
        self.dataset = CustomDataset(self.folder_path, self.window_size)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Train state prediction model')
    parser.add_argument('--motion_dir', type=str, required=True,
                      help='Directory containing motion data')
    parser.add_argument('--batch_size', type=int, default=128,
                      help='Batch size for training')
    parser.add_argument('--history_length', type=int, default=60,
                      help='Length of motion history')
    parser.add_argument('--embedding_dim', type=int, default=3,
                      help='Dimension of motion embedding')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=1000,
                      help='Maximum number of training epochs')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # data_module = MotionDataModule(args.motion_dir, args.history_length, args.batch_size)
    data_module = MotionDataModuleAll4Train(args.motion_dir, args.history_length, args.batch_size)

    # Model training
    model = ComprehensiveModel(
        history_length=args.history_length,
        input_size=316,
        embedding_dim=args.embedding_dim,
        lr=args.lr
    )

    name = f"History{args.history_length}-Embedding{args.embedding_dim}-{os.path.basename(args.motion_dir)}"
    from pytorch_lightning.callbacks import ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'tb_logs/{name}/checkpoints',                 # Specify save path
        filename='{epoch}-{train_loss:.6f}',  # Set filename format
        monitor='train_loss',                      # Monitor training loss
        save_top_k=2,                           # Save best two models
        mode='min'                               # Minimize monitored metric
    )

    # Create Trainer and train model
    tb_logger = TensorBoardLogger(
        save_dir=f'tb_logs/{name}',  # Save logs in 'tb_logs/{name}'
        name='',                     # Avoid creating an extra subdirectory
        version=''                   # Avoid the 'version_0' subdirectory
    )
    trainer = pl.Trainer(
        logger=tb_logger, 
        max_epochs=args.max_epochs, 
        devices=1,
        callbacks=[checkpoint_callback]
        )
    trainer.fit(model, data_module)