from torch.autograd import Variable
import torch
from dataloader import dataloader
import torch.backends.cudnn as cudnn
import os
import numpy as np
import torchvision.models as models

from net import VGG
from weight import extract_weight as ew

# @parameter
# net: the architecture of training network
# dataloader: the set of data which comes from load.py in package dataload
# cost: cost function
# optimizer: the optimization
# epoch: the time of present training process
# n_epochs: the number of training process

def train(net, dataloader, cost, optimizer, epoch, n_epochs, use_cuda):
    # the model of training
    net.train()
    running_loss = 0.0
    print("-" * 10)
    print('Epoch {}/{}'.format(epoch, n_epochs))
    for data in dataloader:
        x_train, y_train = data
        if use_cuda:
            x_train, y_train = x_train.cuda(), y_train.cuda()

        # change data to Variable, a wrapper of Tensor
        x_train, y_train = Variable(x_train), Variable(y_train)

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = net(x_train)
        loss = cost(outputs, y_train)
        loss.backward()
        # optimize the weight of this net
        optimizer.step()

        running_loss += loss.data[0]

    print("Loss {}".format(running_loss / len(dataloader)))
    print("-" * 10)

# @parameter
# net: the architecture of training network
# testloader: the set of data which comes from load.py in package dataload
# cost: cost function

def test(net, testloader, cost, use_cuda):
    net.eval()
    test_loss = 0.0
    correct = 0.0
    total = 0
    print("-" * 10)
    print("test process")

    for data in testloader:
        x_test, y_test = data
        if use_cuda:
            x_test, y_test = x_test.cuda(), y_test.cuda()
        x_test, y_test = Variable(x_test), Variable(y_test)
        output = net(x_test)
        test_loss += cost(output, y_test).data[0]
        _, pred = torch.max(output.data, 1)  # pred: get the index of the max probability
        correct += pred.eq(y_test.data.view_as(pred)).sum()
        total += y_test.size(0)
    print("Loss {}, Acc {}".format(test_loss / len(testloader), 100 * correct / total))
    print("-" * 10)

# def main():
#     use_cuda = torch.cuda.is_available()
#     net = VGG.vgg19()
#     net_root = './netWeight/'
#     print(net)
#
#     if not os.path.exists(net_root):
#         os.mkdir(path=net_root)
#     path = net_root +'vgg19'
#
#     if use_cuda:
#         net.cuda()
#         net = torch.nn.DataParallel(net, device_ids=range(1))
#         cudnn.benchmark = True
#
#     cost = torch.nn.Softmax().cuda()
#
#     if os.path.exists(path):
#         net.load_state_dict(torch.load(path))
#     else:
#         optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#         n_epochs = 250
#         train_set= dataloader.get_train_data()
#         for i in range(n_epochs):
#             train(net=net,
#                   dataloader=train_set,
#                   cost=cost,
#                   optimizer=optimizer,
#                   epoch=i,
#                   n_epochs=n_epochs,
#                   use_cuda=use_cuda)
#         torch.save(net.state_dict(), path)
#         print('successfully save weights')
#
#     test_set = dataloader.get_test_data()
#     test(net=net,testloader=test_set,cost=cost,use_cuda=use_cuda)

def main():
    net = models.vgg19(pretrained=True)
    weight_list = ew.extract_weight(net)
    index,size = ew.attr('D','conv4_1')
    rand_weight_G = ew.generate_random_weight(weight_list, index=index, means=0, std=0.015)
    rand_weight_B = ew.generate_random_weight(weight_list, index=index, means=0, std=1)

    U, s, V = np.linalg.svd(ew.D_matrix(weight_list[index], size, 0), full_matrices=True)
    print(U.shape, V.shape, s.shape)
    np.savetxt("../data/matrixA.csv", s, delimiter=",")

    U, s, V = np.linalg.svd(ew.D_matrix(rand_weight_G, size, 0), full_matrices=True)
    print(U.shape, V.shape, s.shape)
    np.savetxt("../data/matrixB.csv", s, delimiter=",")

    U, s, V = np.linalg.svd(ew.D_matrix(rand_weight_B, size, 0), full_matrices=True)
    print(U.shape, V.shape, s.shape)
    np.savetxt("../data/matrixC.csv", s, delimiter=",")

    # np.savetxt("../data/matrixA.csv", ew.D_matrix(weight_list[index], size, 0), delimiter=",")
    # np.savetxt("../data/matrixB.csv", ew.D_matrix(rand_weight_G, size, 0), delimiter=",")
    # np.savetxt("../data/matrixC.csv", ew.D_matrix(rand_weight_B, size, 0), delimiter=",")


if __name__ == '__main__':
    main()