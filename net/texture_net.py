import torch.nn as nn


class TextureNet(nn.Module):
    def __init__(self):
        super(TextureNet, self).__init__()
        self.name = 'TextureNet'
        self.layers = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=1),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=1),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,384,kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Linear(384,4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,47)

        )
