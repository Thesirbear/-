from torch import nn
from Model.MFM import MFM

class Block(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            input_channels,
            hidden_channels,
            kernel_size = (1, 1),
            stride = 1,
            padding = 1,
        )
        self.mf = MFM(
            input_channels = hidden_channels // 2, 
            hidden_channels = hidden_channels // 2,
            kernel_size = (3, 3)
        )
        self.batch = nn.BatchNorm2d(hidden_channels // 2)
        self.conv2 = nn.Conv2d(
            hidden_channels // 2,
            hidden_channels // 2,
            kernel_size = (3, 3),
            stride = 1,
            padding = 1,
        )

    def forward(self, input):
        input = self.conv1(input)
        input = self.mf(input)
        input = self.batch(input)
        input = self.conv2(input)
        return input