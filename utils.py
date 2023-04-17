import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random

import torchvision
from matplotlib import pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from torchsummary import summary
import cv2

def make_png(att, scale):
    """
    func:
        unsampled the features into 3 channel pic by calculate the mean of the channel
    """
    samlper = nn.UpsamplingBilinear2d(scale_factor=scale)
    att_current = samlper(att)
    att_current = F.relu(att_current, inplace=True)
    att_current = torch.mean(att_current, dim=1)
    att_current = torch.stack([att_current, att_current, att_current], dim=1)
    return att_current


def convert_obj_score(ori_obj_score, MOS):
    """
    func:
        fitting the objetive score to the MOS scale.
        nonlinear regression fit
    """

    def logistic_fun(x, a, b, c, d):
        return (a - b) / (1 + np.exp(-(x - c) / abs(d))) + b

    # nolinear fit the MOSp
    param_init = [np.max(MOS), np.min(MOS), np.mean(ori_obj_score), 1]
    popt, pcov = curve_fit(logistic_fun, ori_obj_score, MOS,
                           p0=param_init, ftol=1e-8, maxfev=2025)
    # a, b, c, d = popt[0], popt[1], popt[2], popt[3]

    obj_fit_score = logistic_fun(ori_obj_score, popt[0], popt[1], popt[2], popt[3])

    return obj_fit_score


def compute_metric(y, y_pred):
    """
    func:
        calculate the sorcc etc
    """
    index_to_del = []
    for i in range(len(y_pred)):
        if y_pred[i] <= 0:
            print("your prediction seems like not quit good, we reconmand you remove it   ", y_pred[i])
            print("The MOS of this pictures is   ", y[i])
            index_to_del.append(i)
    # for i in index_to_del:
    #     y_pred = np.delete(y_pred, i)
    #     y = np.delete(y, i)
    print(y_pred.size)
    print(y.size)
    MSE = mean_squared_error
    RMSE = MSE(y_pred, y) ** 0.5
    PLCC = stats.pearsonr(convert_obj_score(y_pred, y), y)[0]
    # PLCC = stats.pearsonr(y_pred, y)[0]
    SROCC = stats.spearmanr(y_pred, y)[0]
    KROCC = stats.kendalltau(y_pred, y)[0]

    return RMSE, PLCC, SROCC, KROCC


def split_file():
    """
    func:
        split the train dataset and test dataset randomly and
        save it as txt.

    """
    for i in ('5'):
        lines = []
        split_train = open('./data/kon10k/train1000/train_' + str(i), 'a')
        split_test = open('./data/kon10k/test1000/test_' + str(i), 'a')
        with open('./data/kon10k/train.txt', 'r') as infile:
            for line in infile:
                lines.append(line)

        random.shuffle(lines)

        length_train = int(len(lines) * 0.1)
        length_test = int(len(lines) * 0.8)
        split_train.write(''.join(lines[:length_train]))
        split_test.write(''.join(lines[length_test:]))
        split_train.close()
        split_test.close()
    return


def model_parameter(model):
    """
    func:
        use torchsummary to summary the model parameters

    """
    summary(model, input_size=(3, 512, 384), batch_size=-1)
    return


def features_dis(map1, map2, images, bs):
    inv_normalize = torchvision.transforms.Normalize(
        mean=(-2.118, -2.036, -1.804),
        std=(4.367, 4.464, 4.444))
    att_map1 = make_png(map1, 32).permute(0, 2, 3, 1)
    att_map2 = make_png(map2, 32).permute(0, 2, 3, 1)
    images_flip = torch.flip(inv_normalize(images), [3]).permute(0, 2, 3, 1)
    images = inv_normalize(images).permute(0, 2, 3, 1)
    for j in range(bs):
        plt.subplot(2, 2, 1)
        plt.imshow(images[j].cpu().numpy())
        plt.imshow(att_map1[j].cpu().numpy()[:, :, 0], cmap=plt.cm.jet, alpha=0.4)
        # plt.colorbar()
        plt.subplot(2, 2, 2)
        plt.imshow(images_flip[j].cpu().numpy())
        plt.imshow(att_map2[j].cpu().numpy()[:, :, 0], cmap=plt.cm.jet, alpha=0.4)
        plt.subplot(2, 2, 3)
        plt.imshow(images[j].cpu().numpy())
        plt.subplot(2, 2, 4)
        plt.imshow(images_flip[j].cpu().numpy())
        plt.show()
    return


def plot_scatter(y_pred, y_):
    t = np.arctan2(np.array(y_pred), np.array(y_))
    plt.scatter(np.array(y_pred), np.array(y_), alpha=0.5, c=t, marker='.')
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.show()

    return


