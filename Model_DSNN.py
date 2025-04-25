import torch
import torch.nn as nn
import torch.nn.functional as F

class DSNN(nn.Module):
    def __init__(self, input_channels=1, sequence_length=181):
        super(DSNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3))
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(32)
        self.res_conv = nn.Conv2d(16, 32, kernel_size=(1, 1), stride=(1, 4))
        self.dropout = nn.Dropout(0.4)
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self._calculate_conv_output_size()
        self.fc1 = nn.Linear(self.conv_output_size, 64)
        self.fc2 = nn.Linear(64, 6)

    def _calculate_conv_output_size(self):
        x = torch.zeros(1, self.input_channels, 1, self.sequence_length)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        res = self.res_conv(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = x + res
        x = torch.relu(x)
        x = self.dropout(x)
        self.conv_output_size = x.view(1, -1).size(1)
        print(f"Calculated conv_output_size: {self.conv_output_size}")

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        res = self.res_conv(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = x + res
        x = torch.relu(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dequant(x)
        return x