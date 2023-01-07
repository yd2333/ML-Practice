import numpy as np
import torch
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize
from matplotlib import pyplot as plt

def load_MNIST(batch_size, normalize_vals):
    # for correctly download the dataset using torchvision, do not change!
    from six.moves import urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    norm1_val, norm2_val = normalize_vals

    transforms = Compose([ToTensor(), Normalize((norm1_val,), (norm2_val,))])


    train_dataset = torchvision.datasets.MNIST(root='MNIST-data',
                                               train=True,
                                               download=True,
                                               transform=transforms)

    test_dataset = torchvision.datasets.MNIST(root='MNIST-data', 
                                              train=False,
                                              transform=transforms)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)


    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    return train_dataset, test_dataset, train_loader, test_loader

# for plotting the low-dimensional points from the autoencoder in problem 2
def plot_points(points_x, points_y, labels):

    points_x = np.array(points_x)
    points_y = np.array(points_y)
    group = np.array(labels)
    cdict = {0: 'tab:blue',
             1: 'tab:orange',
             2: 'tab:green',
             3: 'tab:red',
             4: 'tab:purple',
             5: 'tab:brown',
             6: 'tab:pink',
             7: 'tab:gray',
             8: 'tab:olive',
             9: 'tab:cyan'}

    fig, ax = plt.subplots()
    for g in np.unique(group):
        ix = np.where(group == g)
        ax.scatter(points_x[ix], points_y[ix], c = cdict[g], label = g, s = 10)
    ax.legend()

    plt.savefig('results.png')
