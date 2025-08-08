import torch
from torch import nn

class MFM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(MFM, self).__init__()

        self.cv1 = nn.Conv2d(
            input_channels, 
            hidden_channels,
            kernel_size = kernel_size,
            padding = 1,
            stride = 1,
        )

        self.cv2 = nn.Conv2d(
            input_channels,
            hidden_channels,
            kernel_size = kernel_size,
            padding = 1,
            stride = 1,
        )

    def forward(self, input):
        x, y = torch.split(input, input.shape[1] // 2, 1)
        x = self.cv1(x)
        y = self.cv2(y)
        return torch.maximum(x, y)

class MFM_fc(nn.Module):
    def __init__(self):
        super(MFM_fc, self).__init__()

    def forward(self, input):
        x, y = torch.split(input, input.shape[1] // 2, 1)
        return torch.maximum(x, y)