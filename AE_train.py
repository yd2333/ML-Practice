import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize
from matplotlib import pyplot as plt
from MyAutoencoder import MyAutoencoder
from GAN_utils import load_MNIST, plot_points

np.random.seed(2022)

batch_size = 10

normalize_vals = (0.1307, 0.3081)

# load MNIST dataset
train_dataset, test_dataset, train_loader, test_loader = load_MNIST(batch_size, normalize_vals)

model = MyAutoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train():
    # training
    epoch = 0
    loss_diff = float('inf')
    train_losses = []
    train_counter = []
    loss = None
    while loss_diff > 0.01 and epoch < 80:
        inepoch_loss = 0
        for images, _ in train_loader:
            # Clear gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model.forward(images)
            # Calculate loss
            loss = criterion(outputs, torch.flatten(images, start_dim=1))
            # Backward pass
            loss.backward()
            # Update weights
            optimizer.step()
            inepoch_loss+=loss.item()
        epoch += 1
        print (f'Epoch [{epoch}], Loss: {inepoch_loss:.4f}')
        if len(train_losses) != 0:
            loss_diff = abs(train_losses[-1]-inepoch_loss)
        train_losses.append(inepoch_loss)

    # plot
    plt.plot(range(1, epoch+1), train_losses)
    plt.show()


    # # save to Path
    # print('Finished Training')
    # PATH = './myae.pth'
    # torch.save(model.state_dict(), PATH)

    # plot
    plt.plot(range(1, epoch+1), train_losses)
    plt.show()

def plot():
    idx = torch.randint(0,6000, (100,)).tolist()
    dataset = train_loader.dataset
    n_samples = len(dataset)

    example_images_x = []
    example_images_y = []
    example_labels = []
    for i, (images, labels) in enumerate(train_loader):
        if i in idx:
            outputs = model.encoder(images)
            example_images_x+=outputs.T.tolist()[0]
            example_images_y+=outputs.T.tolist()[1]
            example_labels+=labels.tolist()
    plt.figure(figsize=(8,6))
    sp_names = list(range(0,10))
    scatter = plt.scatter(example_images_x, example_images_y,
                s=150,
                c=example_labels)
    plt.xlabel("encoded dim 1", size=24)
    plt.ylabel("encoded dim 2", size=24)
    # add legend to the plot with names
    plt.legend(handles=scatter.legend_elements()[0], 
            labels=sp_names,
            title="species")
    # plt.savefig("scatterplot_colored_by_variable_with_legend_matplotlib_Python.png",
    #                     format='png',dpi=150)
    plt.show()


train()
plot()