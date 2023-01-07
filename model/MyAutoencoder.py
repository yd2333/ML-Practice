import numpy as np
import torch
import torch.nn as nn

class MyAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 400, bias=True),
            nn.Tanh(),
            nn.Linear(400, 2, bias=True)
            )

        self.activation = nn.Tanh()

        self.decoder = nn.Sequential(
            nn.Linear(2, 400, bias=True),
            nn.Tanh(),
            nn.Linear(400, 784, bias=True),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.activation(x)
        x = self.decoder(x)
        return  x


