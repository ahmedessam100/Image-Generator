## Importing Libraries
from __future__ import print_function
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as d_set
import torchvision.transforms as trasforms
import torchvision.utils as vision_utils
from altair.vega.v2 import data
from torch.autograd import Variable
from Gen import Generator
from Desc import Descriminator
import torch

## Setting Hyperparameters
batch_size = 64
image_size = 64

## Creating Transformations
transform = trasforms.Compose([trasforms.Scale(image_size), trasforms.ToTensor(), trasforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

## Loading DataSet
data = d_set.CIFAR10(root='./data', download=True, transform = transform)
loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=2)

## Initialzing the weights of input neural network
def weights_init(network):
    class_name = network.__class__.__name__
    if class_name.find('Conv') != -1:
        network.weight.data.normal_(0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        network.weight.data.normal_(1.0, 0.02)
        network.bias.data.fill_(0.0)

# Creating generator object
Gen_network = Generator()
# Initialzing the weights to the network
Gen_network.apply(weights_init)

# Creating Descriminator object
Desc_network = Descriminator()
# Initialzing the weights to the network
Desc_network.apply(weights_init)

## Training The GANs

#Binary Cross Entropy error measure
criteria = nn.BCELoss()
# Descriminatior optimizer
Desc_optimizer = optim.Adam(Desc_network.parameters(), lr=0.0002, betas=(0.5, 0.999))
# Generator optimizer
Gen_optimizer = optim.Adam(Gen_network.parameters(), lr=0.0002, betas=(0.5, 0.999))

epochs = 25

for epoch in range(epochs):
    for idx, data_mini in enumerate(loader, 0):

        # Updating The weights of the descriminator
        Desc_network.zero_grad()

        # Training descriminator on batches of real images
        re_image, labels = data_mini
        input = Variable(re_image)
        target = Variable(torch.ones(input.size()[0]))
        re_output = Desc_network(input)
        error_Desc_re = criteria(re_output, target)

        # Training Descriminator on batches of fake images
        noise_signal = Variable(torch.randn(input.size()[0], 100, 1, 1))
        fake_image = Gen_network(noise_signal)
        target = Variable(torch.zeros(input.size()[0]))
        fake_output = Desc_network(fake_image.detach())
        error_Desc_fake = criteria(re_output, target)

        # Backpropagating the total error
        total_error = error_Desc_fake + error_Desc_re
        total_error.backward()
        Desc_optimizer.step()

        # Updating The weights of the Generator
        Gen_network.zero_grad()
        target = Variable(torch.ones(input.size()[0]))
        Gen_output = Desc_network(fake_image)
        Gen_error = criteria(Gen_output, target)
        Gen_error.backward()
        Gen_optimizer.step()

        # Printing the losses and saving the real and generated images
        print('[%d/%d][%d/%d] Descriminator Loss : %.4f , Generator Loss : %.4f'%(epoch, epochs, idx, len(loader), total_error.data[0], Gen_error.data[0]))

        # Save the image every 100 step
        if (idx % 100) == 0:
            #Save Real images
            vision_utils.save_image(re_image, './result/real_samples.png', normalize=True)
            fake_im = Gen_network(noise_signal)
            vision_utils.save_image(fake_im.data, './result/fake_samples_epochs_%03d.png' % (epoch), normalize=True)


