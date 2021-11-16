import machineLearnCode.Utils.shiro_unilt as su
import math
import numpy as np

def create_decision_tree(data, feature_name_list):
    """
    构造决策树
    采用递归的方法，逐步划分数据集，生成节点存储，构造决策树
    :param data:                训练集，nparray，最后一列为标签列
    :param feature_name_list:   每一列特征的名字
    :return:                    { 根节点：{子节点1：{子子节点1}, 子节点2:{ class1 }} }
    """

    # 递归停止的条件
    # 1.当前数据集中所有分类相同.
    try:
        current_class = [temp[-1] for temp in data[:, -1:]]
        ndim = data.shape[1]
    except:
        current_class = [temp[-1] for temp in data]
        ndim = 1
    if current_class.count(current_class[0]) == len(current_class):
        return current_class[0]

    # 2.当数据集无法再分割，即只剩下最后一列特征时
    if ndim <= 2:
        # 由多数来表决结点。【这里就可能会出现分类错误的问题】
        max_count = 0
        max_count_class = ''
        for cla in current_class:
            cla_count = current_class.count(cla)
            if cla_count >= max_count:
                max_count = cla_count
                max_count_class = cla
        return max_count_class

    # 不满足停止条件，继续递归
    # 选出当前数据集中，最适合用来当作节点的属性列序号，这里采用改进后的ID3算法
    best_column = chooseBestNode(data)
    best_column_label = feature_name_list[best_column]
    decisionTree = {best_column_label : {}}
    # todo 如果是连续型变量，则不删除，该列能参与下一次划分。
    # todo 以下是针对离散型变量的，需要补充对连续型变量的处理
    del(feature_name_list[best_column])
    # 然后以当前最佳feature去划分
    best_column_value = set([values[best_column] for values in data])
    for value in best_column_value:
        print(f'当前划分列为：{best_column_label}，划分属性为：{value}')
        sub_label = feature_name_list[:]  # 这里是深拷贝一份特征标签
        # 当前结点的子节点
        decisionTree[best_column_label][value] = create_decision_tree(
            split_data_by_unique_value(data, best_column, value),
            sub_label
        )

    return decisionTree

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
        shannonEnt -= p * math.log(p, 2)

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
    # split_result_list每一个元素都是一个特征的npArray，然后对这个List遍历计算每个属性的信息增益与熵
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



def CART_classify(data, column_name_list):
    """
    CART分类树，使用基尼指数
    :param data:    nparray数据集 ，最后一列是标签列
    :param column_name_list:  nparray列名
    :return:
    """
    split_feature = ''
    split_value = ''
    split_column = 0
    min_gini = 99999
    size_c = len(column_name_list)
    for column in range(size_c):
        # 遍历当前数据集，对每个特征逐一遍历，找到Gini指数最小的目标特征与目标值，作为划分依据
        cur_feature_gini, cur_feature_value = get_feature_Gini(data, column)
        if cur_feature_gini < min_gini:
            split_feature = cur_feature_gini
            split_value = cur_feature_value
            split_column = column

    if split_feature != '':
        # todo 划分数据集
        print('划分数据集')

def get_feature_Gini(data, column):
    """
    循环计算目标 特征列下每个值 的基尼指数
    取最小的那个作为该特征的划分
    :param data:    二维数组、矩阵等
    :param column:  当前划分的 feature 的index
    :return:
    """
    value_list = set(data[:,column].tolist())
    total_count = len(value_list)   # 数据集总数
    min_gini = 99999
    split_value = ''
    for cur_value in value_list:
        # 将data划分为 =cur_value 和 !=cur_value的两个子集
        equal_part = []     # =cur_value
        unequal_part = []   # !=cur_value
        for row in data:
            if cur_value == row[column]:
                equal_part.append(row)
            else :
                unequal_part.append(row)
        # 分别计算两个子集的gini
        equal_gini = get_value_Gini(equal_part)
        unequal_gini = get_value_Gini(unequal_part)
        cur_value_gini = equal_gini * len(equal_part) / total_count + unequal_gini * len(unequal_part) / total_count
        if cur_value_gini < min_gini :
            min_gini = cur_value_gini
            split_value = cur_value
    return min_gini, split_value

def get_value_Gini(data):
    """
    计算目标 特征列下，划分后子集的 基尼系数
    :param data: 二维数组、矩阵等
    :return: Gini
    """
    label_count = {}  # 标签计数{ label1:count, }
    total_count = len(data)  # 总数
    cur_l = 0  # 不让下方出现警告提示
    for row in data:
        cur_l = row[-1]
        if cur_l not in label_count.keys():
            label_count[cur_l] = 0
        label_count[cur_l] += 1

    label_size = len(label_count)
    if label_size > 2:
        # 多分类
        return calculate_Gini_more(total_count, label_count)
    elif label_size == 2:
        # 二分类
        return calculate_Gini_two(total_count, label_count[cur_l])
    else :
        # 当前数据集只有一个类别
        return 0



def calculate_Gini_more(total_count, label_count):
    """
    多分类时的基尼指数计算，1 - sum(pi^2)
    :param total_count:
    :param label_count:
    :return:
    """
    p = 0.0
    for label in label_count:
        p += (label_count[label] / total_count) ** 2

    return 1-p

def calculate_Gini_two(total_count, label_count):
    """
    二分类时的计算， 2·p·（1-p）
    :param total_count: 总数量
    :param label_count: 其中一个标签的计数
    :return:
    """
    p = label_count / total_count
    return 2 * p * (1-p)
