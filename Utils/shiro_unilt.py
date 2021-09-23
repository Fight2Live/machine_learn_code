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

def get_shannon_Ent(data):
    """
    计算香农熵

    :param data: array矩阵，其中最后一列为标签
    :return: norm_data：array矩阵
    """
    numEntries = len(data)
    label_count = {}
    # 先计算数据集中各类型数量
    for temp in data:
        label = temp[-1]
        if label not in label_count.keys():
            label_count[label] = 0
        label_count[label] += 1

    # 计算香农熵
    shannonEnt = 0.0
    for label in label_count:
        p = float(label_count[label]) / numEntries
        shannonEnt -= p * log(p, 2)

    return shannonEnt

def split_data_by_shannon(data, axis, value):
    data_set = []
    for feat_vec in data:
        if feat_vec[axis] == value:
            reduced = feat_vec[:axis]
            reduced.extend(feat_vec[axis+1:])
            data_set.append(reduced)

def expected_information(data):
    """
    计算期望信息 I
    :param data: ndarray，最后一列是标签列
    :return:
    """
    class_count = {}  # 标签计数
    sum_count = len(data)
    for item in data:
        # 计算各标签的数量
        current_class = item[-1]
        if current_class not in class_count.keys():
            class_count[current_class] = 0
        class_count[current_class] += 1

    expectI = 0.0  # 期望信息 I
    for class_temp in class_count:
        p = class_count[class_temp] / sum_count
        expectI -= p * math.log(p, 2)

    return expectI


def get_comentropy(data):
    """
    计算目标属性的信息熵
    :param data: 元素A，和类别C构成的二维矩阵 ndarray
    :return:
    """
    sum_count = len(data)
    element_array = {}  # 每个元素值的对应矩阵
    # 将属性A列，根据其下不同的值去划分成长度为J的字典，Key是属性A的值j，Value是属性A = j时的数据矩阵
    for item in data:
        element = str(item[0])
        if element not in element_array:
            element_array[element] = item
        else :
            element_array[element] = np.vstack((element_array[element], item))

    # 计算信息熵 E（A）
    E = 0.0
    for element in element_array:
        I = expected_information(element_array[element])
        E += len(element_array[element]) / sum_count * I

    return E

def chooseBestNode(data):
    """
    选择最好的节点，即信息熵最大的一列。
    但因为信息熵是由 总数据集的信息熵 - 目标列信息熵 得到的，这里没有去算总信息熵，所以取反，即取最小值
    :param data:
    :return:
    """
    # 将高维数据集按列拆成长度为L的List，L是数据集的属性数 + 1（标签列）
    # split_result_list每一个元素都是一个属性npArray，然后对这个List遍历计算每个属性的信息增益与熵
    split_result_list = np.split(data, data.shape[1], axis=1)
    element_arrays = split_result_list[:-1]
    # 其中最后一个元素是标签列
    label_arrays = split_result_list[-1]
    entropy = []
    best_E = 999999
    best_element = ''
    for index in range(len(element_arrays)):
        element_array = element_arrays[index]
        target_array = np.hstack((element_array, label_arrays))
        E = get_comentropy(target_array)
        entropy.append(E)
        if E < best_E:
            best_E = E
            best_element = index
    print('信息熵结算结果：{}'.format(entropy))
    print('最佳节点的元素列为：{}'.format(best_element) )
    return best_element



def split_data_by_unique_value(data, index, value):
    """
    切割并组合成新矩阵返回。
    将数据集中第index列，值为value的行凑为一个新矩阵
    :param data:    nparray
    :param index:   要删除的第X列
    :return:
    """
    t2 = 0
    for item in data:
        if item[index] == value:
            t1 =  np.hstack((item[:index], item[index + 1 :]))
            if t2 == 0:
                t2 = t1
            else:
                t2 = np.vstack((t2, t1))
    print('划分后的数据集：{}'.format(t2))
    return t2





if __name__ == '__main__':
    pass