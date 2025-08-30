import logging
import os
import sys
import time
from typing import Dict, List, Tuple
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants and configuration
CONFIG = {
    'model': 'LSTM',
    'num_epochs': 10,
    'batch_size': 32,
    'learning_rate': 0.001,
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.2,
    'log_dir': 'logs',
    'data_dir': 'data',
    'model_dir': 'models',
}

class AgentDataset(Dataset):
    def __init__(self, data: pd.DataFrame, seq_len: int):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx: int):
        seq = self.data.iloc[idx:idx + self.seq_len]
        x = seq[['feature1', 'feature2', 'feature3']].values
        y = seq['target'].values
        return {'x': x, 'y': y}

class AgentModel(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float):
        super(AgentModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class AgentTrainer:
    def __init__(self, model: AgentModel, device: torch.device, config: Dict):
        self.model = model
        self.device = device
        self.config = config
        self.optimizer = Adam(self.model.parameters(), lr=config['learning_rate'])
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)
        self.writer = SummaryWriter(log_dir=config['log_dir'])

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        for epoch in range(self.config['num_epochs']):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                x, y = batch['x'].to(self.device), batch['y'].to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = nn.MSELoss()(output, y.view(-1, 1))
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            self.writer.add_scalar('train_loss', total_loss / len(train_loader), epoch)
            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                predictions = []
                labels = []
                for batch in val_loader:
                    x, y = batch['x'].to(self.device), batch['y'].to(self.device)
                    output = self.model(x)
                    loss = nn.MSELoss()(output, y.view(-1, 1))
                    val_loss += loss.item()
                    predictions.extend(output.cpu().numpy())
                    labels.extend(y.cpu().numpy())
                self.writer.add_scalar('val_loss', val_loss / len(val_loader), epoch)
                self.writer.add_scalar('val_acc', accuracy_score(labels, np.round(predictions)), epoch)
                self.writer.add_scalar('val_f1', f1_score(labels, np.round(predictions)), epoch)
                self.scheduler.step(accuracy_score(labels, np.round(predictions)))
            logger.info(f'Epoch {epoch+1}, Train Loss: {total_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}')

    def evaluate(self, test_loader: DataLoader):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            predictions = []
            labels = []
            for batch in test_loader:
                x, y = batch['x'].to(self.device), batch['y'].to(self.device)
                output = self.model(x)
                loss = nn.MSELoss()(output, y.view(-1, 1))
                total_loss += loss.item()
                predictions.extend(output.cpu().numpy())
                labels.extend(y.cpu().numpy())
            logger.info(f'Test Loss: {total_loss / len(test_loader)}')
            logger.info(f'Test Acc: {accuracy_score(labels, np.round(predictions))}')
            logger.info(f'Test F1: {f1_score(labels, np.round(predictions))}')

def main():
    # Load data
    data = pd.read_csv(os.path.join(CONFIG['data_dir'], 'data.csv'))
    scaler = StandardScaler()
    data[['feature1', 'feature2', 'feature3']] = scaler.fit_transform(data[['feature1', 'feature2', 'feature3']])
    train_data, val_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_dataset = AgentDataset(train_data, seq_len=10)
    val_dataset = AgentDataset(val_data, seq_len=10)
    test_dataset = AgentDataset(test_data, seq_len=10)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    # Create model and trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AgentModel(input_dim=3, hidden_size=CONFIG['hidden_size'], num_layers=CONFIG['num_layers'], dropout=CONFIG['dropout'])
    model.to(device)
    trainer = AgentTrainer(model, device, CONFIG)

    # Train model
    trainer.train(train_loader, val_loader)

    # Evaluate model
    trainer.evaluate(test_loader)

if __name__ == '__main__':
    main()