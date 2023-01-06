import numpy as np

import torch
import torch.nn as nn

class MyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(128, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256,512,bias=True),
            nn.ReLU(),
            nn.Linear(512,1024,bias=True),
            nn.ReLU(),
            nn.Linear(1024,784,bias=True),
            nn.Tanh()
        )
        

    def forward(self, x):
        return self.block(x)

