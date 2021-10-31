import decisionTree
import random
import numpy as np
from sklearn.linear_model import LinearRegression

def create_random_forest(data, label, k, p):
    """
    构造随机森林
    :param data:    总样本集，nparray
    :param label:   特征列名列表
    :param k:       生成多少颗树
    :param p:       抽取样本集的p%作为训练集，1-p%作为测试集
    :return:        [decision1, decision2,  ]
    """
    data_size = data.shape[0]           # 总样本集个数
    simple_size = int(data_size * p)    # 训练集个数



    final_forest = []
    for poll in range(k):
        # 需要生成多少棵树，就循环几次
        data_temp = data[:, :]
        # 从样本集总随机抽取simple_size个作为训练集
        nn = data_temp[random.randint(0, data_size)]
        for count in range(simple_size - 1):
            nn = np.vstack((nn, data_temp[random.randint(0, data_size-1)]))
        label_temp = label[:]
        final_forest.append(decisionTree.create_decision_tree(nn, label_temp))

    return final_forest

