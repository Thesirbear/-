from torch import nn
import torch

from Model.Block import Block

from Model.MFM import MFM
from Model.MFM import MFM_fc

class LCNN(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_of_samples, output_size):
        super().__init__()

        self.conv1 = nn.Conv2d(
            input_channels,
            hidden_channels,
            kernel_size = (5, 5),
            padding = 1,
            stride = 1,
        )

        self.Block1 = Block(
            hidden_channels // 2,
            hidden_channels // 2,
        )
        self.Block2 = Block(
            hidden_channels // 4,
            hidden_channels // 4,
        )
        self.Block3 = Block(
            hidden_channels // 8,
            hidden_channels // 8,
        )
        self.Block4 = Block(
            hidden_channels // 16,
            hidden_channels // 16,
        )
        self.mxpool1 = nn.MaxPool2d(
            kernel_size = (2, 2),
            stride = 2,
        )
        
        self.mxpool2 = nn.MaxPool2d(
            kernel_size = (2, 2),
            stride = 2,
        )
        self.mxpool3 = nn.MaxPool2d(
            kernel_size = (2, 2),
            stride = 2,
        )
        self.mxpool4 = nn.MaxPool2d(
            kernel_size = (2, 2),
            stride = 2,
        )
        
        self.batch1 = nn.BatchNorm2d(hidden_channels // 4)
        self.batch2 = nn.BatchNorm2d(hidden_channels // 16)
        self.batch3 = nn.BatchNorm1d(10)
        self.mfm1 = MFM(
            hidden_channels // 2, 
            hidden_channels // 2,
            kernel_size = (3, 3)
        )
        self.mfm2 = MFM_fc()
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear((hidden_channels // 32) * 192, 20)
        self.drop = nn.Dropout(0.1)
        self.soft = nn.Softmax(dim=1)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mfm1(x)
        
        x = self.mxpool1(x)
        
        x = self.Block1(x)
        x = self.mxpool2(x)
        x = self.batch1(x)


        x = self.Block2(x)
        x = self.mxpool3(x)
    
        x = self.Block3(x)
        x = self.batch2(x)
        
        x = self.Block4(x)
        x = self.mxpool4(x)
        
        x = self.flat(x)
        x = self.fc1(x)
        x = self.mfm2(x)
        x = self.drop(x)
        x = self.batch3(x)
        x = self.fc2(x)
        x = self.soft(x)
        x = torch.split(x, x.shape[1] // 2, 1)[0]
        x = torch.squeeze(x, dim=1)
        return x
