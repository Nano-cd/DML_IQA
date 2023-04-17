import os
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image
from scipy.signal import convolve2d
import numpy as np
import math

data_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize([512, 384]),
])


def default_loader(path):
    return Image.open(path).convert('L')  #


def LocalNormalization(patch, P=3, Q=3, C=1):
    kernel = np.ones((P, Q)) / (P * Q)
    patch_mean = convolve2d(patch, kernel, boundary='symm', mode='same')
    patch_sm = convolve2d(np.square(patch), kernel, boundary='symm', mode='same')
    patch_std = np.sqrt(np.maximum(patch_sm - np.square(patch_mean), 0)) + C
    patch_ln = torch.from_numpy((patch - patch_mean) / patch_std).float().unsqueeze(0)
    return patch_ln


def OverlappingCropPatches(im, patch_size=32, stride=32):
    w, h = im.size
    centerx = int(w / 2) + 1
    centery = int(h / 2) + 1
    vecenter = np.array([centerx, centery])
    patches_dis = ()
    distance = ()
    entropy = ()
    for i in range(0, h - stride, stride):
        for j in range(0, w - stride, stride):
            imm = im.crop((j, i, j + patch_size, i + patch_size))
            patch = to_tensor(imm)
            patch_dis = LocalNormalization(patch[0].numpy())
            x1 = j + 1
            y1 = i + 1
            vec1 = np.array([x1, y1])
            x2 = j + 1
            y2 = i + 32
            vec2 = np.array([x2, y2])
            x3 = j + 32
            y3 = i + 1
            vec3 = np.array([x3, y3])
            x4 = j + 32
            y4 = i + 32
            vec4 = np.array([x4, y4])
            x_cen = int(x4 / 2) + 1
            y_cen = int(y4 / 2) + 1
            vec_cen = np.array([x_cen, y_cen])
            dist1 = np.linalg.norm(vec1 - vecenter)
            dist2 = np.linalg.norm(vec2 - vecenter)
            dist3 = np.linalg.norm(vec3 - vecenter)
            dist4 = np.linalg.norm(vec4 - vecenter)
            dist5 = np.linalg.norm(vec_cen - vecenter)
            dist = torch.tensor([[[dist1, dist2, dist3, dist4, dist5]]])

            patches_dis = patches_dis + (patch_dis,)
            distance = distance + (dist,)
            entropyy = ComputerEntropy(imm)
            entropy = entropy + (entropyy,)

    return patches_dis, distance, entropy


def ComputerEntropy(im):
    tmp = []
    for i in range(256):
        tmp.append(0)
    val = 0
    k = 0
    res = 0
    img = np.array(im)
    for i in range(len(img)):
        for j in range(len(img[i])):
            val = img[i][j]
            tmp[val] = float(tmp[val] + 1)
            k = float(k + 1)
    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)
    for i in range(len(tmp)):
        if (tmp[i] == 0):
            res = res
        else:
            res = float(res - tmp[i] * (np.log2(tmp[i])))
    res = torch.tensor([[[res]]])
    return res


def OverlappingCropPatches_gradient(im, patch_size=32, stride=32):
    w, h = im.size
    patches_gradient = ()
    for i in range(0, h - stride, stride):
        for j in range(0, w - stride, stride):
            patch = to_tensor(im.crop((j, i, j + patch_size, i + patch_size)))
            patches_gradient = patches_gradient + (patch,)
    return patches_gradient


class IQADataset(Dataset):
    def __init__(self, label_file, image_root, grad_root):
        self.patch_size = 32
        self.stride = 32
        self.imagepaths = []
        self.gradpaths = []
        self.labels = []
        self.img = []
        self.gradiant_img = []
        self.patches_dis = ()
        self.patches_gradient = ()
        self.distance = ()
        self.entropy = ()
        self.label = []
        self.label_std = []

        self.scale = data_transform
        with open(label_file, 'r') as f:
            for line in f.readlines():  # 读取label文件
                self.imagepaths.append(os.path.join(image_root, line.split()[1]))
                self.gradpaths.append(os.path.join(grad_root, line.split()[1]))
                self.labels.append(float(line.split()[0]))
        # range(len(self.imagepaths))
        for idx in range(len(self.imagepaths)):
            im = self.scale(default_loader(self.imagepaths[idx]))
            im_gradient = self.scale(default_loader(self.gradpaths[idx]))
            self.img.append(im)
            self.gradiant_img.append(im_gradient)
            patches_dis, distance, entropy = OverlappingCropPatches(im, self.patch_size, self.stride)
            patches_gradient = OverlappingCropPatches_gradient(im_gradient, self.patch_size, self.stride)
            self.patches_dis = self.patches_dis + patches_dis
            self.patches_gradient = self.patches_gradient + patches_gradient
            self.distance = self.distance + distance
            self.entropy = self.entropy + entropy
            for i in range(len(patches_dis)):
                self.label.append(self.labels[idx])
                self.label_std.append(self.labels[idx])

    def __len__(self):
        return len(self.patches_dis)

    def __getitem__(self, idx):
        return ((self.patches_dis[idx], self.patches_gradient[idx], self.distance[idx], self.entropy[idx]),
                (torch.Tensor([self.label[idx]])))

