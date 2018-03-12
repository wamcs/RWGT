
import torch
import numpy as np
from torch.autograd import Variable

vggtypes = {'A': ['conv1_1', 'conv2_1', 'conv3_1', 'conv3_2', 'conv4_1', 'conv4_2', 'conv5_1', 'conv5_2'],
            'B': ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv4_1', 'conv4_2', 'conv5_1',
                  'conv5_2'],
            'C': ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2',
                  'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3'],
            'D': ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4', 'conv4_1',
                  'conv4_2', 'conv4_3', 'conv4_4', 'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4']}

vgg_size = {'A': [226, 114, 58, 58, 30, 30, 16, 16, 7, 1, 1],
            'B': [226, 226, 114, 114, 58, 58, 30, 30, 16, 16, 7, 1, 1],
            'C': [226, 226, 114, 114, 58, 58, 58, 30, 30, 30, 16, 16, 16, 7, 1, 1],
            'D': [226, 226, 114, 114, 58, 58, 58, 58, 30, 30, 30, 30, 16, 16, 16, 16, 7, 1, 1]}


# extract the weight value of convolution layer in model
def extract_weight(model):
    weight_list = []
    for i, param in enumerate(model.parameters()):
        if i % 2 == 0:
            weight_list.append(param)
    return weight_list


# generate the random weight which has the same size with appointed layer using normal distribution
# @parameter
# weight_list: the list of convolution kernel coming from function extract_weight
# index: the index of appointed layer
# means: means of normal distribution
# std:  The standard deviation of normal distribution
def generate_random_weight(weight_list, index, means, std):
    temp = weight_list[index]
    size = 1
    for i in temp.size():
        size = size * i
    ran_weight = torch.normal(torch.FloatTensor([means] * size), torch.FloatTensor([std] * size))
    ran_weight = Variable(ran_weight.view(list(temp.size())))
    return ran_weight


# generate the 2d fourier transform base
# let f is the 1d fourier transform base, which comes from fft(eye(n)) / sqrt(n)
# we need is f*f, operation * is kronecker product
def twoD_fourier_base(size):
    print('-' * 10)
    print('begin to generate base')
    base = np.fft.fft(np.eye(size)) / np.sqrt(size)
    base = np.matrix(base)
    print('finish')
    print('-' * 10)
    return np.kron(a=base, b=base)


# generate the convolution matrix
# if the size of image is n. when it is vectored as a n*n vector, size of the matrix we need is (n*n,n*n)
def generate_conv_matrix(conv_kernel, size):
    print('-' * 10)
    print('begin to generate convoution matrix')
    # image is (n*n,1) vector, concatenating it's lines all
    conv = np.matrix(conv_kernel.data.numpy())
    conv = conv.T
    conv_size = conv.shape[0]

    # generate part circulant block
    temp_list = []
    for weight in conv:
        temp_list.append(np.pad(weight, ((0, 0), (0, size - conv_size)), mode='constant', constant_values=(0, 0)))

    print(np.array(temp_list).shape)
    temp_part_circulant_block = []
    for item in temp_list:
        temp = []
        for i in range(size):
            temp.append(np.roll(item, shift=i))
        temp_part_circulant_block.append(np.concatenate(tuple(temp), axis=0))

    print(np.array(temp_part_circulant_block).shape)
    part_circulant_block = np.concatenate(tuple(temp_part_circulant_block), axis=1)
    part_circulant_block = np.pad(part_circulant_block, ((0, 0), (0, size * (size - conv_size))), mode='constant',
                                  constant_values=(0, 0))
    print(part_circulant_block.shape)
    # generate circulant block

    result = []
    for k in range(size):
        result.append(np.roll(part_circulant_block, shift=k * size, axis=1))

    temp = np.concatenate(tuple(result), axis=0)
    print(temp.shape)
    print('finish')
    print('-' * 10)
    return np.matrix(temp)


def test_is_dial(matrix):
    size1 = matrix.shape[0]
    size2 = matrix.shape[1]
    if size1 != size2:
        raise TypeError("matrix is not n*n matrix")
    a = np.linalg.det(matrix)
    b = np.diag(matrix)
    c = 1
    for i in b:
        c = c * i
    print(a, c)
    if not (a == c):
        raise TypeError("matrix is not diagonal")


def part_D(conv_kernel, size, base):
    print('-' * 10)
    print('begin to generate part D matrix')
    conv = generate_conv_matrix(conv_kernel=conv_kernel, size=size)
    D = np.dot(base.I, conv)
    D = np.dot(D, base)
    # print(D)
    # test_is_dial(D)
    print('finish')
    print('-' * 10)
    return np.diag(D)

# @parameter:
# weight: the filter weights of layer
# channel: if channel is -1, then deal all the filter
def D_matrix(weight, size, channel=0):
    D = []
    base = twoD_fourier_base(size=size)
    for item in weight:
        temp = item[channel]
        part_D_matrix = part_D(temp, size, base)
        D.append(part_D_matrix)

    result = []
    for i, valueI in enumerate(D):
        for j, valueJ in enumerate(D):
            if j < i:
                continue
            result.append(np.matrix(valueI * valueJ))

    temp = np.concatenate(tuple(result), axis=0)
    print(temp.shape)
    return temp


def attr(type,layer):
    index = vggtypes[type].index(layer)
    size = vgg_size[type][index]
    return index,size
