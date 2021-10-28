import machineLearnCode.Utils.shiro_unilt as su


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
    current_class = [temp[-1] for temp in data[:, -1:]]
    if current_class.count(current_class[0]) == len(current_class):
        return current_class[0]

    # 2.当数据集无法再分割，即只剩下最后一列特征时
    if data.shape[1] <= 2:
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
    best_column = su.chooseBestNode(data)
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
            su.split_data_by_unique_value(data, best_column, value),
            sub_label
        )

    return decisionTree




