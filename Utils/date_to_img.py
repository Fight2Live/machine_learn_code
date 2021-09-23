import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn import datasets
import itertools

from .shiro_unilt import *

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def scatter_diagram(data, label):
    """
    散点图
    :param label:
    :param data:
    :return:
    """
    # 所传入的数据的属性个数，两两组合后的结果
    attr_count = data[0].shape[0]
    plt.figure()
    # 将每个属性两两组合后的散列图展示出来
    col_pairs = itertools.combinations(range(attr_count), 2)  # itertools.combinations(x, y) 对x进行排列，每个结果有y个元素
    col_pairs = list(col_pairs)

    img_size = get_img_size(len(col_pairs))
    subplot_start = int(str(img_size[1]) + str(img_size[0]) + '1')
    print(subplot_start)

    for i in col_pairs:
        plt.subplot(subplot_start)
        plt.scatter(data[:,i[0]], data[:, i[1]], 15.0*np.array(label), 15.0*np.array(label))  # 参数c为点设置颜色，不同类别花的点颜色不同
        plt.xlabel('第 {} 列数据'.format(str(i[0])))
        plt.ylabel('第 {} 列数据'.format(str(i[1])))
        # 坐标轴上的值
        # plt.xticks([])
        # plt.yticks([])
        subplot_start += 1
    plt.show()

def pca_scatter_diagram(data, label):
    """
    将经过PCA降维后的数据展示出来
    :param data:
    :param label:
    :return:
    """
    # 数据降维器
    tsne = TSNE(n_components=2, init='pca', random_state=1)
    result = tsne.fit_transform(data)
    x_min, x_max = np.min(result), np.max(result)

    # 这一步似乎让结果都变为0-1的数字
    result = (result - x_min) / (x_max - x_min)
    fig = plt.figure()
    # subplot可以画出一个矩形，长宽由参数的前两位确定，参数越大，边长越小
    ax = plt.subplot(111)
    for i in range(result.shape[0]):
        plt.text(result[i, 0], result[i, 1], str(label[i]), color=plt.cm.Set1(int(label[i]) / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title('hello')
    plt.show(fig)

def get_line_chart(x, y, label, xyLabel):
    """
    这里是绘画折线图
    :param x: list
    :param y: list
    :param label: list
    :param xyLabel: {xLabel, yLabel}
    :return:
    """
    plt.figure()
    for index in range(len(y)):
        plt.plot(x, y[index], label=label[index])
    plt.xlabel(xyLabel['xLabel'])
    plt.ylabel(xyLabel['yLabel'])
    plt.legend(loc='upper right')
    plt.show()



