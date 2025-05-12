from isaacgym.torch_utils import *
from skillmimic.utils import torch_utils

import os
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader, random_split


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
        return x

class ComprehensiveModel(pl.LightningModule):
    # def __init__(self, history_length):
    #     super(ComprehensiveModel, self).__init__()
    #     self.history_encoder = HistoryEncoder(history_length)
    #     self.fc1 = nn.Linear(3 + 316 + 64, 512)
    #     self.fc2 = nn.Linear(512, 256)
    #     self.fc3 = nn.Linear(256, 316)
    def __init__(self, history_length, input_size=394, embedding_dim=3, lr=0.001):
        super(ComprehensiveModel, self).__init__()
        self.save_hyperparameters()
        self.history_encoder = HistoryEncoder(history_length, input_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim + input_size + 64, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 394)

    def forward(self, history, current_motion, current_label):
        history_features = self.history_encoder(history)
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
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        history, current_motion, current_label, y = batch
        y_hat = self(history, current_motion, current_label)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.parameters(), lr=0.001)
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5),
            'monitor': 'train_loss',  # Monitor validation loss for scheduling
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
            if f.endswith('.pt')
        ]
        # self.file_paths = [os.path.join(motion_dir, f) for f in os.listdir(motion_dir)]
        
        print(f'############################ Have load {len(self.file_paths)} motions ############################')
        self.data = []
        
        for file_path in self.file_paths:
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
                self.data.append((history_data[i], current_motion_data[i], current_label_data, target_data[i]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def data_to_state(self, data):
        ###### data ######
        # pt data (409 dim): root_pos(3) + root_rot(3) + root_rot(3) + dof_pos(60*3) + body_pos(71*3) 
        #                   + obj_pos(3) + zero_obj_rot(3) + contact_graph(1)
        ###### state ######
        # humanoid_obs (823 dim): root_h (1) + local_body_pos (52*3) + local_body_rot (53*6) 
        # + local_body_vel (53*3) + local_body_ang_vel (53*3) + body_contact_buf (30)
        # obj_obs (15 dim): local_tar_pos (3) + local_tar_rot (6) + local_tar_vel (3) + local_tar_ang_vel (3)
        nframes = data.shape[0]
        root_pos = data[:, :3]
        root_rot = data[:, 3:6]
        body_pos = data[:, 189:189+71*3].reshape(nframes, 71, 3)
        humanoid_obs = compute_humanoid_observations(root_pos, root_rot, body_pos) # (nframes, 211)
        humanoid_obs = torch.cat((humanoid_obs, data[:, 9:9+180]), dim=-1) # (nframes, 391)
        obj_pos = data[:, 402:405] # (nframes, 3)
        obj_obs = compute_obj_observations(root_pos, root_rot, obj_pos) # (nframes, 3)
        state = torch.cat((humanoid_obs, obj_obs), dim=-1) # (nframes, 394)
        # # 5% noise
        # noise_flag = np.random.rand() < 0.05
        # if noise_flag:
        #     state = state + torch.randn_like(state) * 0.05
        return state

class MotionDataModule(pl.LightningDataModule):
    def __init__(self, folder_path, window_size, batch_size=32):
        super().__init__()
        self.folder_path = folder_path
        self.window_size = window_size
        self.batch_size = batch_size

    def setup(self, stage=None):
        # 在此处初始化数据集并进行划分
        dataset = CustomDataset(self.folder_path, self.window_size)
        
        # 随机划分训练集和验证集
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
        # 在此处初始化数据集并进行划分
        self.dataset = CustomDataset(self.folder_path, self.window_size)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

# Usage
motion_dir = 'skillmimic/data/motions/Sigraph_experiment/app3/parahome_mix/'
history_length = 60
custom_dataset = CustomDataset(motion_dir, history_length)
print('Dataset Length:', len(custom_dataset))
dataloader = DataLoader(custom_dataset, batch_size=256, shuffle=True)

# TensorBoard Logger
logger = TensorBoardLogger("output/tb_logs", name="Parahome")

# Model training
model = ComprehensiveModel(history_length)
trainer = pl.Trainer(max_epochs=3000, devices=1, logger=logger)
trainer.fit(model, dataloader)

### Command ###
# CUDA_VISIBLE_DEVICES=1 python state_prediction_local_parahome_ry.py
###############
