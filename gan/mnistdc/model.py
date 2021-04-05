import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, negative_slope=0.2):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.MaxPool2d((2, 2))
        )

        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 1024),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.dis(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class Generator(nn.Module):
    def __init__(self, input_size, out_width=28):
        super(Generator, self).__init__()
        self.out_width = out_width
        self.first_layer_width = out_width * 2
        self.fc = nn.Linear(input_size, self.first_layer_width * self.first_layer_width)
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        self.gen = nn.Sequential(
            nn.Conv2d(1, 50, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU(inplace=True),
            nn.Conv2d(50, 25, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU(True),
            nn.Conv2d(25, 1, kernel_size=2, stride=2),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.fc(x)
        out = out.view(-1, 1, self.first_layer_width, self.first_layer_width)
        out = self.br(out)
        out = self.gen(out)
        return out

