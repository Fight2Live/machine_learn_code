from Utils.shiro_unilt import *
from Utils.date_to_img import *
from Classification.KNN.my_knn import *

#from ..shiro_unilt import *
import pandas as pd
import numpy as np
import time
import operator

def get_sample_data_by_txt(file_path):
    """
    从txt中获取样本集数据
    :return:
    """
    f = open(file_path)
    file_data = f.readlines()
    sample = []
    label = []
    # 处理换行符，并以\t分割数据
    for index in range(len(file_data)):
        item = file_data[index].strip().replace('\n', '').split('\t')
        sample.append([float(j) for j in item[:-1]])
        try:
            label.append(int(item[-1]))
        except:
            label.append(item[-1])

    # 转矩阵
    sample = np.array(sample)

    return sample, label

def test_accurate_rate(k):
    """
    测试K_NN的精确率
    :return:
    """
    sample, label = get_sample_data_by_txt(fp)
    # 取前80%作为训练集
    trans_data = sample[: int(len(sample) * 0.8)]
    trans_label = label[: int(len(label) * 0.8)]
    # 剩下的为测试集
    test_data = sample[int(len(sample) * 0.8) : ]
    test_label = label[int(len(label) * 0.8) : ]

    # 数据归一化
    trans_data = auto_norm(trans_data)
    test_data = auto_norm(test_data)
    accurate_rate = []
    start = time.time()
    for kk in range(1,k+1):
        correct_count = 0
        error_count = 0
        for i in range(len(test_data)):
            predict = test_data[i]
            predict_class = my_knn(predict, trans_data, trans_label, kk)
            if predict_class == test_label[i]:
                correct_count += 1
            else:
                error_count += 1

        #print('正确个数：{}\n错误个数：{}'.format(correct_count, error_count))
        current_rate = correct_count / (correct_count + error_count) * 100
        accurate_rate.append(current_rate)
        #print('K：{}，精确率：{}%'.format(kk, current_rate))
    end = time.time()
    print('K的范围：（1~{}），精确率计算耗时：{}秒'.format(k, end-start))

    return accurate_rate

def test_accuracy(k):
    """
        测试K_NN的准确率
        :return:
        """
    sample, label = get_sample_data_by_txt(fp)
    sample_data = auto_norm(sample)
    accuracy_rate = []
    start = time.time()
    for kk in range(1, k+1):
        correct_count = 0
        error_count = 0
        for i in range(len(sample_data)):
            predict = sample_data[i]
            predict_class = my_knn(predict, sample_data, label, kk)
            if predict_class == label[i]:
                correct_count += 1
            else:
                error_count += 1

        # print('正确个数：{}\n错误个数：{}'.format(correct_count, error_count))
        current_rate = correct_count / (correct_count + error_count) * 100
        accuracy_rate.append(current_rate)
        #print('K：{}，准确率：{}%'.format(kk, current_rate))
    end = time.time()
    print('K的范围：（1~{}），准确率计算耗时：{}秒'.format(k, end - start))

    return accuracy_rate

def export_resurt(k):
    """
    以精确率最高时的K值去进行分类，并将结果作为表格导出
    :param k:
    :return:
    """
    # 获取样本集和对应的标签矩阵
    sample, label = get_sample_data_by_txt(fp)
    class_result_list = []
    # 样本集数据归一化
    sample_data = auto_norm(sample)

    for index in range(len(sample_data)):
        target_dat = sample_data[index]
        class_result_list.append(my_knn(target_dat, sample_data, label, k))  # 保存分类结果

    # 将分类结果与原数据拼接，方便导出
    old_label = np.array([label])
    class_result_np = np.array([class_result_list])

    export_data = np.hstack((np.hstack((sample, old_label.T)), class_result_np.T))
    # print(export_data)
    np.savetxt('./KNN分类结果（K={}）.csv'.format(k), export_data, delimiter=',',
               header='特征一,特征二,特征三,原标签,分类结果' ,
               fmt=('%f, %f, %f, %d, %d'))


def get_rate_img(k):
    accurate_rate = test_accurate_rate(k)
    accuracy_rate = test_accuracy(k)
    x = range(1,k+1)
    y = [accurate_rate, accuracy_rate]
    label = ['精确率', '准确率']
    xyLabel = {'xLabel':'K值',
               'yLabel':'百分比'}

    get_line_chart(x, y, label, xyLabel)



fp = './DataSet/datingTestSet2.txt'
cur_K = 2
# 精确率
test_accurate_rate(cur_K)

# 准确率
test_accuracy(cur_K)

# 画图
# sample, label = get_sample_data_by_txt(fp)
# # 对特征进行PCE降维后的散点图
# scatter_diagram(sample, label)
# pca_scatter_diagram(sample, label)
# # K从1-20时，样本的精确率与准确率变化情况
# get_rate_img(20)
#
# # 导出
# export_resurt(cur_K)


