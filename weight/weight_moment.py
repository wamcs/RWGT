import numpy as np
import torch
import torchvision.models as models
from weight import extract_weight as ew
import numpy as np

net = models.vgg19(pretrained=True)
print(net)
weight_list = ew.extract_weight(net)

result = {}
for i,weight in enumerate(weight_list):
    print(i)
    temp_result = []
    temp = weight.data
    size = 1
    for i in weight.size():
        size = size * i
    temp = temp.view(size, -1)
    s1 = torch.sum(temp)
    temp_result.append(s1)
    print(s1)

    temp2 = torch.mul(temp, temp)
    s2 = torch.sum(temp2)
    temp_result.append(s2)
    print(s2)

    temp3 = torch.mul(temp2, temp)
    s3 = torch.sum(temp3)
    temp_result.append(s3)
    print(s3)

    temp4 = torch.mul(temp2, temp2)
    s4 = torch.sum(temp4)
    temp_result.append(s4)
    print(s4)
    print(size)
    result[size] = np.array(temp_result)

num = 0
sum = np.array([0, 0, 0, 0])
for n in result.keys():
    num += n
    sum = sum + result[n]

print(num)
print(sum)
temp_moment = sum / num

print('one moment is {}'.format(temp_moment[0]))
print('two moment is {}'.format(temp_moment[1] - temp_moment[0] ** 2))
print('three moment is {}'.format(temp_moment[2] - 3 * temp_moment[1] * temp_moment[0] + 2 * (temp_moment[0] ** 3)))
print('four moment is {}'.format(
    temp_moment[3] - 4 * temp_moment[2] * temp_moment[0] + 6 * temp_moment[1] * (temp_moment[0] ** 2) - 3 * (
    temp_moment[0] ** 4)))
