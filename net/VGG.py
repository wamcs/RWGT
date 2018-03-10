import torch.nn as nn
import torch.nn.functional as F

# this vgg is not the original version, it is created as ranVGG which is mentioned in
# paper A Powerful Generative Model Using Random Weights for the Deep Image Representation

# maximal pooling function is changed to average pooling function

vggtypes = {'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                      'M']}





class VGG(nn.Module):
    def __init__(self, vggtype):
        super(VGG, self).__init__()
        self.name = vggtype
        self.layer = self.generate_layer(vggtype)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            self.activation_function(),
            nn.Linear(4096, 4096),
            self.activation_function(),
            nn.Linear(4096, 1000)
        )

    def generate_layer(self, type):
        layers = []
        in_channels = 3
        for i in vggtypes[type]:
            if i == 'M':
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, i, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(i))
                layers.append(self.activation_function())
                in_channels = i
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def name(self):
        return self.name

    def activation_function(self):
        return nn.ReLU(inplace=True)

    def construction(self,type):
        count = 1
        in_count = 1
        name_layers = ['input data']
        for i in vggtypes[type]:
            if i != 'M':
                name_layers.append('conv' + str(count) + '_' + str(in_count))

                in_count += 1
            else:
                name_layers.append('pool' + str(count))
                in_count = 1
                count += 1
        print(name_layers)


def vgg11():
    return VGG('VGG11')


def vgg13():
    return VGG('VGG13')


def vgg16():
    return VGG('VGG16')


def vgg19():
    return VGG('VGG19')
