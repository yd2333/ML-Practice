import numpy as np
import torch
import torch.nn as nn

class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 1-channel input, size 28x28
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3,stride=1, padding=0, dilation=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout(p=0.5)
        )
        
        self.flatten = nn.Flatten()

        self.linear_block = nn.Sequential(
            nn.Linear(20*13*13, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10, bias=True)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.flatten(x)
        x = self.linear_block(x)
        return x

