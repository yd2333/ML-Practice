import numpy as np
import torch
import torch.nn as nn

class MyDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(784,1024,bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.block(x)