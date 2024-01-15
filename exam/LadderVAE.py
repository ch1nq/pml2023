import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

class LadderVAE(nn.Module):
    def __init__(self):
        super(LadderVAE, self).__init__()

        # BottomUp
        #self.x_in = nn.Linear(in_features=784, out_features=400) # X input
        self.d_1 = nn.Linear(in_features=784, out_features=392) # Deterministic 1
        self.d_2 = nn.Linear(in_features=392, out_features=196) # Deterministic 2
        self.d_1 = nn.Conv2d(in_channels=1,
                            out_channels=2,
                            kernel_size=3,
                            padding=1,
                            stride=2,
                            groups=1)
        self.d_2 = nn.Conv2d(in_channels=2,
                            out_channels=4,
                            kernel_size=3,
                            padding=1,
                            stride=2,
                            groups=1)

        self.bup_out1 = nn.Linear(in_features=392, out_features=2) # BottomUp Out 1
        self.bup_out2 = nn.Linear(in_features=392, out_features=2) # BottomUp Out 2

        # TopDown
        self.tdown1 = nn.Linear(in_features=2, out_features=196) # TopDown 1
        self.z_2 = nn.Linear(in_features=196, out_features=392) # Stochastic 2
        self.z_1 = nn.Linear(in_features=392, out_features=400) # Stochastic 1

        self.x_out = nn.Linear(in_features=400, out_features=784) # X i output

    def encode(self, x):
        #x_in = F.relu(self.x_in(x))
        bsize = x.shape[0]
        #print(x.shape)
        x = x.reshape((bsize, 1, 28, 28))
        d_1 = F.relu(self.d_1(x)) # Conv2d
        #print('d1: ',d_1.shape)
        z_1 = F.relu(self.z_1(d_1.reshape((bsize, 392))))

        d_2 = F.relu(self.d_2(d_1))
        #print('d_2: ',d_2.shape)
        d_2 = d_2.reshape((bsize, 196))

        z_2 = F.relu(self.z_2(d_2))
        z_1_2 = F.relu(self.z_1(z_2))

        return self.bup_out1(z_2), self.bup_out2(z_2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        tdown1 = F.relu(self.tdown1(z))
        z_2 = F.relu(self.z_2(tdown1))
        z_1 = F.relu(self.z_1(z_2))
        #h3 = F.relu(self.fc3(z))
        #h4 = F.relu(self.fc3a(h3))
        
        #h3 = F.relu(self.fc22(z))
        #h4 = F.relu(self.fc3a(h3))

        return torch.sigmoid(self.x_out(z_1))

    def forward(self, x): # when we call model(data), it is the same as model.forward(data)
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar