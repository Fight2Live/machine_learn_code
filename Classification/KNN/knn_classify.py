import numpy as np


def knn_classify(predict, sample, label, k):
    """
    Knn分类算法
    :param predict:     待分类样数据，单条数据，array
    :param sample:      训练样本集的特征元素矩阵，array
    :param label:       训练样本集的标签集，list
    :param k:
    :return:
    """

    # 样本数
    sample_size = sample.shape[0]
    # 将待分类的数据由一维数组，扩充为sample_size维数组
    # np.tile(a, (b,c)) 将a重复b行c列
    predict_data = np.tile(predict, (sample_size, 1))
    # 使用欧氏距离计算
    distances = (((predict_data - sample) ** 2).sum(axis = 1)) ** 0.5
    print(f'欧氏距离计算结果：{distances}')
    # 对距离进行排序，从小到大
    sort_distances_result = distances.argsort()
    # 对距离最近的前k个标签进行记数
    class_count = {}
    for i in range(k):
        try:
            class_count[label[sort_distances_result[i]]] += 1
        except:
            class_count[label[sort_distances_result[i]]] = 1

    # 对记数结果进行从大到小的排序，返回分类结果list
    class_result = sorted(class_count, reverse=True)
    return class_result[0]



