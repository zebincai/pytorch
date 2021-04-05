import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, negative_slope=0.2):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(784, 300),
            nn.LeakyReLU(negative_slope),
            nn.Linear(300, 150),
            nn.LeakyReLU(negative_slope),
            nn.Linear(150, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.dis(x)
        return out


class Generator(nn.Module):
    def __init__(self, input_size):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(input_size, 150),
            nn.ReLU(True),
            nn.Linear(150, 300),
            nn.ReLU(True),
            nn.Linear(300, 784),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.gen(x)
        return x

