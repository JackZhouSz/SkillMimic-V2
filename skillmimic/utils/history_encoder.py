import torch
import torch.nn as nn
import pytorch_lightning as pl

class HistoryEncoder(nn.Module):
    def __init__(self, history_length, input_dim=330, output_dim=32, final_dim=3):
        super(HistoryEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=output_dim, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(output_dim * history_length, final_dim)

        self.dropout = nn.Dropout(p=0.2)
    
    def resume_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        # Extract the model's state dict
        state_dict = checkpoint['state_dict']
        # Remove the prefix 'history_encoder.' and load into the current model
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('history_encoder.'):
                new_key = k.replace('history_encoder.', '')
                new_state_dict[new_key] = v
        self.load_state_dict(new_state_dict)


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
    def __init__(self):
        super(ComprehensiveModel, self).__init__()
        self.history_encoder = HistoryEncoder()
        self.fc1 = nn.Linear(3 + 330 + 64, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 330)

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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
