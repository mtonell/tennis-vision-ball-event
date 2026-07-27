import torch
import torch.nn as nn

class TrajectoryCNN(nn.Module):
    def __init__(self, sequence_length, in_channels=2, num_classes=2):
        super(TrajectoryCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.flatten = nn.Flatten()
        
        flat_size = 64 * (sequence_length // 2)
        self.fc1 = nn.Linear(flat_size, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TrajectoryLSTM(nn.Module):
    def __init__(self, sequence_length, in_channels=2, num_classes=2, hidden_size=64, num_layers=2, dropout=0.3):
        super(TrajectoryLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

class TrajectoryBiLSTM(nn.Module):
    def __init__(self, sequence_length, in_channels=2, num_classes=2):
        super(TrajectoryBiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(64, num_classes) # 32 hidden_size * 2 directions
        
    def forward(self, x):
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        # Use mean pooling over the sequence for BiLSTM to capture both directions equally
        return self.fc(out.mean(dim=1))

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(TCNBlock, self).__init__()
        padding = (3 - 1) * dilation // 2 
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu_out = nn.ReLU()
        
    def forward(self, x):
        out = self.dropout1(self.relu1(self.bn1(self.conv1(x))))
        out = self.dropout2(self.relu2(self.bn2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)

class TrajectoryTCN(nn.Module):
    def __init__(self, sequence_length, in_channels=2, num_classes=2):
        super(TrajectoryTCN, self).__init__()
        
        self.block1 = TCNBlock(in_channels, 32, dilation=1)
        self.block2 = TCNBlock(32, 64, dilation=2)
        self.block3 = TCNBlock(64, 64, dilation=4)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

class TrajectoryTransformer(nn.Module):
    def __init__(self, sequence_length, in_channels=2, num_classes=2):
        super(TrajectoryTransformer, self).__init__()
        self.embedding = nn.Linear(in_channels, 32)
        self.pos_embedding = nn.Parameter(torch.zeros(1, sequence_length, 32))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(32, num_classes)
        
    def forward(self, x):
        x = x.transpose(1, 2) # (batch, seq_len, features)
        x = self.embedding(x)
        x = x + self.pos_embedding # Position information
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class TrajectoryResNet(nn.Module):
    def __init__(self, sequence_length, in_channels=2, num_classes=2):
        super(TrajectoryResNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        
        self.layer1 = ResidualBlock1D(32, 32)
        self.layer2 = ResidualBlock1D(32, 64, stride=2)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

class InceptionModule1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule1D, self).__init__()
        bottleneck_channels = out_channels // 4
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1)
        
        self.conv3 = nn.Conv1d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(bottleneck_channels, bottleneck_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(bottleneck_channels, bottleneck_channels, kernel_size=7, padding=3)
        
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.pool_conv = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1)
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        bot = self.bottleneck(x)
        p1 = self.conv3(bot)
        p2 = self.conv5(bot)
        p3 = self.conv7(bot)
        p4 = self.pool_conv(self.pool(x))
        
        out = torch.cat([p1, p2, p3, p4], dim=1)
        return self.relu(self.bn(out))

class TrajectoryInception(nn.Module):
    def __init__(self, sequence_length, in_channels=2, num_classes=2):
        super(TrajectoryInception, self).__init__()
        self.initial = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.bn_initial = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        
        self.inc1 = InceptionModule1D(32, 64)
        self.inc2 = InceptionModule1D(64, 128)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.relu(self.bn_initial(self.initial(x)))
        x = self.inc1(x)
        x = self.inc2(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

def get_dl_model(model_name, sequence_length, in_channels=2, num_classes=2):
    if model_name == 'cnn':
        return TrajectoryCNN(sequence_length, in_channels, num_classes)
    elif model_name == 'lstm':
        return TrajectoryLSTM(sequence_length, in_channels, num_classes)
    elif model_name == 'bilstm':
        return TrajectoryBiLSTM(sequence_length, in_channels, num_classes)
    elif model_name == 'tcn':
        return TrajectoryTCN(sequence_length, in_channels, num_classes)
    elif model_name == 'transformer':
        return TrajectoryTransformer(sequence_length, in_channels, num_classes)
    elif model_name == 'resnet':
        return TrajectoryResNet(sequence_length, in_channels, num_classes)
    elif model_name == 'inception':
        return TrajectoryInception(sequence_length, in_channels, num_classes)
    else:
        raise ValueError(f"Unknown DL model architecture: {model_name}")
