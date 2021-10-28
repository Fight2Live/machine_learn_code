import decisionTree
import matplotlib.pyplot as plt
import numpy as np

"""
绘制决策树
"""

def getNumLeafs(tree):
    """
    获取叶子节点数目
    :param tree:
    :return:        叶子数
    """
    num_leafs = 0
    first_str = list(tree.keys())[0]
    second_dict = tree[first_str]
    for key in second_dict:
        if type(second_dict[key]).__name__ == 'dict':
            num_leafs += getNumLeafs(second_dict[key])
        else:
            num_leafs += 1

    return num_leafs

def getTreeDepth(tree):
    """
    获取树的深度
    :param tree:
    :return:        深度
    """
    max_depth = 0
    first_str = list(tree.keys())[0]
    second_dict = tree[first_str]
    for key in second_dict:
        if type(second_dict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(second_dict[key])
        else:
            thisDepth = 1

        if thisDepth > max_depth :
            max_depth = thisDepth

    return max_depth


def plotNode(nodeTxt, conterPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt,
                            xy = parentPt,
                            xycoords='axes fraction',
                            xytext = conterPt,
                            textcoords = 'axes fraction',
                            va="center",
                            ha = "center",
                            bbox = nodeType,
                            arrowprops = arrow_args)

def plotMidText(cntrPt, parentPt, txtString):
    """
    在父子节点间填充文本信息
    :param cntrPt:
    :param parentPt:
    :param txtString:
    :return:
    """
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(tree, parentPt, nodeTxt):
    """
    计算宽与高
    :param tree:
    :param parentPt:
    :param nodeTxt:
    :return:
    """
    numLeafs = getNumLeafs(tree)
    depth = getTreeDepth(tree)
    firstStr = list(tree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 /plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = tree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in list(secondDict.keys()):
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD

def createPlot(tree):
    fig = plt.figure(1, facecolor = 'white')
    fig.clf()
    axprops = dict(xticks = [], yticks = [])
    createPlot.ax1 = plt.subplot(111, frameon = False, **axprops)
    plotTree.totalW = float(getNumLeafs(tree))
    plotTree.totalD = float(getTreeDepth(tree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(tree, (0.5, 1.0), '')
    plt.show()

if __name__ == '__main__':
    decisionNode = dict(boxStyle="sawtooth", fc="0.8")
    leafNode = dict(boxstyle="round4", fc="0.8")
    arrow_args = dict(arrowstyle="<-")

    data = np.genfromtxt(r'./lenses.txt', delimiter='\t', dtype=str)
    label = ['age', 'prescript', 'astigmatic', 'tearRate']

    decisionTree = decisionTree.create_decision_tree(data, label)
    print(decisionTree)
    createPlot(decisionTree)
