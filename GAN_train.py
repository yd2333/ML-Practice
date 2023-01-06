import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize
from matplotlib import pyplot as plt
from MyGenerator import MyGenerator
from MyDiscriminator import MyDiscriminator
from GAN_utils import load_MNIST

np.random.seed(2022)

batch_size = 128

normalize_vals = (0.5, 0.5)

# load MNIST dataset
train_dataset, test_dataset, train_loader, test_loader = load_MNIST(batch_size, normalize_vals)


# init
gen = MyGenerator()
dis = MyDiscriminator()
optimizer_gen = torch.optim.Adam(gen.parameters(), lr=0.0002)
optimizer_dis = torch.optim.Adam(dis.parameters(), lr=0.0002)
criterion_gen = nn.BCELoss()
criterion_dis = nn.BCELoss()


# train
def train_and_plot_loss():
    # train
    epoch = 0
    loss_diff = float('inf')
    train_losses_gen = []
    train_losses_dis = []
    while loss_diff > 0.01 and epoch < 40:
        loss_dis = 0
        loss_gen = 0
        epoch+=1

        for images, _ in train_loader:
            
            optimizer_gen.zero_grad()
            optimizer_dis.zero_grad()

            noise = torch.normal(0, 1, size=(batch_size,128))
            images_fake = gen(noise)

            # discriminator making judging
            judge_real = dis(images.flatten(1))
            judge_fake = dis(images_fake)

            # loss: discriminator
            loss_r = criterion_dis(judge_fake, torch.zeros_like(judge_fake))
            loss_f = criterion_dis(judge_real, torch.ones_like(judge_real))
            loss_sum = loss_r + loss_f
            loss_sum.backward(retain_graph=True)
            optimizer_dis.step()
            loss_dis += loss_sum.item()
            
            # loss: gen
            output = dis(images_fake)
            loss = criterion_gen(output, torch.ones_like(judge_fake))
            loss.backward()
            optimizer_gen.step()
            loss_gen += loss.item()
            
        
        print (f'Epoch [{epoch}], Discriminator Loss: {loss_dis:.4f}, Generator Loss: {loss_gen:.4f}')
        if len(train_losses_dis) != 0:
            loss_diff = abs(train_losses_dis[-1]-loss_dis+train_losses_gen[-1]-loss_gen)
        train_losses_dis.append(loss_dis)
        train_losses_gen.append(loss_gen)
        # generate pic after epoch
        fig, axs = plt.subplots(5)
        fig.suptitle(f"iter{epoch}")
        for i in range(5):
            noise = torch.normal(0, 1, size=(128,))
            images_fake = gen(noise).reshape((28,28))
            axs[i].imshow(images_fake.detach().numpy())
        plt.savefig(f"./pics/generated{epoch}.png")

    
    # save to Path
    # print('Finished Training')
    # PATH = './mygenerator.pth'
    # torch.save(gen.state_dict(), PATH)
    # PATH = './mydiscriminator.pth'
    # torch.save(dis.state_dict(), PATH)
    # plot
    plt.plot(range(1, epoch+1), train_losses_dis, label= "Discriminator Loss")
    plt.plot(range(1, epoch+1), train_losses_gen, label= "Generator Loss")
    plt.legend(loc="upper left")
    plt.savefig(f"./pics/Loss.png")
            

            
            
train_and_plot_loss()