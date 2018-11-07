import torch.nn as nn

## Defining The Generator
class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        # Creating the model of the deconvolutional neural network
        self.model = nn.Sequential(
                    # It takes input random vector of size 100
                    nn.ConvTranspose2d(in_channels=100, out_channels=512, kernel_size=4, stride=1, padding=0,
                                       bias=False),
                    nn.BatchNorm2d(num_features=512),   # Normalize the features according to the dimension of the batch
                    nn.ReLU(True),
                    nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(num_features=256),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(num_features=128),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(num_features=64),
                    nn.ReLU(True),
                    # We are making 3 channels
                    nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
                    # For non-linearity and to specify values ranges from -1 to 1 centered at 0
                    nn.Tanh()
                    )
    # Function that forwards random signal of vector of size 100 to the network represents some noise to create the fake image
    def forward(self, *input):
         # Returning the output consists of 3 channels
          return self.model(*input)
