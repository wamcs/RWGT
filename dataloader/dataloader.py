import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import os

path = '/home/data/data/ImageNet_ilsvrc2012_2014/'

transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                             std = [ 0.229, 0.224, 0.225 ]),
    ])
def get_train_data():
    traindir = os.path.join(path, 'train')

    train = datasets.ImageFolder(traindir, transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=512, shuffle=True, num_workers=4)

    return train_loader

def get_test_data():
    valdir = os.path.join(path, 'val')
    val = datasets.ImageFolder(valdir, transform)
    val_loader = torch.utils.data.DataLoader(val, batch_size=512, shuffle=True, num_workers=4)
    return val_loader