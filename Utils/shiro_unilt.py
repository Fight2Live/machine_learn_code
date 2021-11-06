from numpy import *
import math
import numpy as np

def get_img_size(num):
    """
    求一个数的所有因素
    同时最后的一个结果可以作为plt.subplot的size
    :param num:
    :return:
    """
    start = 1
    end = num
    factor = []
    while start < end:
        if start * end == num:
            factor.append([start, end])
            start += 1
        end -= 1
        if start * end < num:
            start += 1
    return factor[-1]


def auto_norm(data):
    """
    将数值归一化
    newValue = (oldValue - min)/(max-min)
    :param data: array矩阵
    :return: norm_data：array矩阵
    """
    min_values = data.min(0)
    max_values = data.max(0)
    ranges = max_values - min_values

    indexs = data.shape[0]
    norm_data = data - tile(min_values, (indexs, 1))
    norm_data = norm_data / tile(ranges, (indexs, 1))

    return norm_data

def mean_norm(data, mean, std):
    """
    将数值归一化（均值方差标准化）
    newValue = (oldValue - mean)/标准差
    :param data: nparray矩阵
    :param mean: 平均值
    :param std: 标准差
    :return: norm_data：nparray矩阵
    """

    indexs = data.shape[0]
    norm_data = data - tile(mean, (indexs, 1))
    norm_data = norm_data / tile(std, (indexs, 1))

    return norm_data







if __name__ == '__main__':
    pass