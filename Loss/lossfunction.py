import numpy as np
from PIL import Image
import os

root_path = '../dtd/images'
save_path = '../data/dtd'
s = 112

paths = os.listdir(root_path)


def DFT_base(size=s):
    base = np.fft.fft(np.eye(size)) / np.sqrt(size)
    base = np.matrix(base)
    return base


def DFT(img, base, two_dim_base):
    data = np.matrix(img)
    temp = np.dot(base, data)
    temp = np.dot(temp, base)

    temp = temp.T.flatten().T
    para = np.dot(two_dim_base.I, temp)
    print('*')
    return np.array(para.T)


base = DFT_base()
two_dim_base = np.kron(base, base)

if not os.path.exists(save_path):
    os.makedirs(save_path)

for path in paths:
    eigns = []
    for file in os.listdir(root_path + '/' + path):
        img = Image.open(root_path + '/' + path + '/' + file)
        img = img.resize((s, s))
        img = img.convert('L')
        para = DFT(img, base, two_dim_base)
        eigns.append(para)
    eigns = np.array(eigns)
    np.savetxt(save_path + '/' + path + '.csv', eigns, delimiter=',')
