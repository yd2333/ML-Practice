import numpy as np
import torch
import torch.nn as nn

import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize

from matplotlib import pyplot as plt

from MyCNN import MyCNN

from GAN_utils import load_MNIST

np.random.seed(2022)

batch_size = 32

normalize_vals = (0.1307, 0.3081)

# load MNIST dataset
train_dataset, test_dataset, train_loader, test_loader = load_MNIST(batch_size, normalize_vals)

# Training
model = MyCNN()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
criterion = nn.CrossEntropyLoss()
epoch = 0
loss_diff = float('inf')
train_losses = []
train_counter = []
loss = None
while loss_diff > 0.001 and epoch < 1000:
    inepoch_loss = 0
    epoch += 1

    for i, (images, labels) in enumerate(train_loader):
        # Clear gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model.forward(images)
        # Calculate loss
        loss = criterion(outputs, labels)
        # Backward pass
        loss.backward()
        # Update weights
        optimizer.step()
        inepoch_loss+=loss.item()
    print (f'Epoch [{epoch}], Loss: {inepoch_loss:.4f}')
    if len(train_losses) != 0:
        loss_diff = abs(train_losses[-1]-inepoch_loss)
    train_losses.append(inepoch_loss)

# save the model
# print('Finished Training')
# PATH = './mycnn001.pth'
# torch.save(model.state_dict(), PATH)

# plot the training loss
plt.plot(range(1, epoch+1), train_losses)
plt.show()


# Testing
test_loss = 0
n_correct = 0
n_samples = 0
for images, labels in test_loader:
    outputs = model(images)
    loss = criterion(outputs, labels)
    test_loss += loss.item()
    # max returns (value ,index)
    _, predicted = torch.max(outputs, 1)
    n_samples += labels.size(0)
    n_correct += (predicted == labels).sum().item()

acc = 100.0 * n_correct / n_samples
print(f'Accuracy of the network: {acc} %')
print(f'Loss of the network: {test_loss} %')




