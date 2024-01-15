from sklearn.metrics import mean_squared_error  # RMSE
from sklearn.metrics import r2_score            # 決定係数
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np

import pickle
from .formated import format_data
import pandas as pd
from .common import MUSIC_FEATURE
import os

file_path = os.path.dirname(os.path.realpath(__file__))

"""
[37, 34, 41] があり、クラス 0 のサンプルが 37 個、クラス 1 のサンプルが 34 個、クラス 2 のサンプルが 41 個あることを示しています。
"""


def get_rules(clf):
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    values = clf.tree_.value

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            print(node_id, depth ,stack)
            print()
            is_leaves[node_id] = True

    print(
        "The binary tree structure has {n} nodes and has "
        "the following tree structure:\n".format(n=n_nodes)
    )

    count_leaf = 0
    
    for i in range(n_nodes):
        if is_leaves[i]:
            print(
                "{space}node={node} is a leaf node with value={value}.".format(
                    space=node_depth[i] * "\t", node=i, value=values[i]
                )
            )
            count_leaf += 1
        else:
            print(
                "{space}node={node} is a split node with value={value}: "
                "go to node {left} if X[:, {feature}] <= {threshold} "
                "else to node {right}.".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    left=children_left[i],
                    feature=MUSIC_FEATURE[feature[i]],
                    threshold=threshold[i],
                    right=children_right[i],
                    value=values[i],
                )
            )

    
    def add_path(i, path):
        path += str(i)+","
        if children_left[i]==children_right[i]:
            path_str_index_list.append(path)
            return
        add_path(children_left[i], path)
        add_path(children_right[i], path)
    
    path_str_index_list = []
    add_path(0, "")

    path_index_lsit = [p[:-1].split(',') for p in path_str_index_list]

    path_list = []

    for path_idxs in path_index_lsit:
        path = []
        for p in path_idxs:
            p = int(p)
            obj = {"value": values[p] , 
                "condition": "go to node {left} if X[:, {feature}] <= {threshold} else to node {right}.".format(
                    left=children_left[p],
                    feature=MUSIC_FEATURE[feature[p]],
                    threshold=threshold[p],
                    right=children_right[p],
                    value=values[p],
                ), 
                "depth":node_depth[p],
                "node":p,
                "is_leaf": is_leaves[p],
                }
            path.append(obj)
        path_list.append(path)
    
    print(path_list[-1])

    










# RandomForestの分類器を作る
def random_forest_classifier_maker(data):
    df = format_data(data)

    X = df.loc[:, MUSIC_FEATURE].values
    y = df["rank"]

    # ランダムフォレスト回帰
    # forest = RandomForestClassifier(random_state=1234)
    forest = RandomForestClassifier(random_state=1234, n_estimators=5)
    # モデル学習
    forest.fit(X, y)

    # 学習モデルの保存
    with open(file_path+'/models/randomForestModel.pickle', mode='wb') as f:
        pickle.dump(forest, f, protocol=2)

    """
    # 推論
    y_train_pred = forest.predict(X)

    # 平均平方二乗誤差(RMSE)
    print('RMSE 学習: %.2f' % (
        mean_squared_error(y, y_train_pred, squared=False)  # 学習
    ))
    # 決定係数(R^2)
    print('R^2 学習: %.2f' % (
        r2_score(y, y_train_pred)  # 学習
    ))

    # Feature Importance
    fti = forest.feature_importances_
    print(fti)
    """
    print("random forest")
    # print(forest.estimators_)
    # print(forest.estimators_[0].tree_)
    # t = forest.estimators_[0]
    # tree.plot_tree(t)
    # plt.show()

    get_rules(forest.estimators_[0])


# 実際にRandomForestで分類する
# THINK:現状引数にリストを渡さないといけないので、オブジェクト1つでもできるように
# サイズ１の[{hogehoge}]が渡されてくることを想定
def classify_data_by_random_forest(data):
    # モデルのオープン
    with open(file_path+'/models/randomForestModel.pickle', mode='rb') as f:
        forest = pickle.load(f)

    df = pd.DataFrame(data)

    X = df.loc[:, MUSIC_FEATURE].values

    result = forest.predict(X)

    if result[0]:
        return 1
    else:
        return 0


def get_random_forest_importance():
    # モデルのオープン
    with open(file_path+'/models/randomForestModel.pickle', mode='rb') as f:
        forest = pickle.load(f)

    # Feature Importance
    fti = forest.feature_importances_
    importance_abs_dic = dict()
    for i in range(len(fti)):
        importance_abs_dic[MUSIC_FEATURE[i]] = abs(fti[i])
    importance_abs_dic = sorted(
        importance_abs_dic.items(), key=lambda x: x[1], reverse=True)

    return importance_abs_dic
