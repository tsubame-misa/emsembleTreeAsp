from sklearn.metrics import mean_squared_error  # RMSE
from sklearn.metrics import r2_score            # 決定係数
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

import pickle
from .formated import format_data
import pandas as pd
from .common import MUSIC_FEATURE
import os

file_path = os.path.dirname(os.path.realpath(__file__))


def get_evaluation(path, X, y, df):
    # そのpathを通ったXについて
    predict_y = []
    ans_y = []
    predicted_x_index = []

    for index, row in df.iterrows():
        leaf = True
        for i in range(len(path)):
            condition = path[i]["condition"]
            if condition["left"] ==  path[i]["next"]:
                if not row[condition["feature"]] <= condition["threshold"]:
                    leaf = False if i!=len(path)-1 else True
                    break  
            else:
                if not row[condition["feature"]] > condition["threshold"]:
                    leaf = False if i!=len(path)-1 else True
                    break
        if leaf:
            if condition["value"][0][0] > condition["value"][0][1]:
                predict_y.append(0)
            else:
                predict_y.append(1)
            ans_y.append(y[index])
            predicted_x_index.append(index)

    if len(predict_y) > 0:
        eval = {"accuracy":accuracy_score(y_true=ans_y, y_pred=predict_y),
                "precision": precision_score(y_true=ans_y, y_pred=predict_y),
                "recall":recall_score(y_true=ans_y, y_pred=predict_y),
                "f1":f1_score(y_true=ans_y, y_pred=predict_y)}
        return eval

    return {"accuracy": None,"precision":  None,"recall": None,"f1": None}





"""
[37, 34, 41] があり、クラス 0 のサンプルが 37 個、クラス 1 のサンプルが 34 個、クラス 2 のサンプルが 41 個あることを示しています。
"""

def get_rules(clf, X, y, df):
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
            is_leaves[node_id] = True

    print(
        "The binary tree structure has {n} nodes and has "
        "the following tree structure:\n".format(n=n_nodes)
    )
    
    def add_path(i, path):
        path += str(i)+","
        path_str_index_list.append(path)
        if children_left[i]==children_right[i]:
            return
        add_path(children_left[i], path)
        add_path(children_right[i], path)
    
    path_str_index_list = []
    add_path(0, "")

    path_index_lsit = [p[:-1].split(',') for p in path_str_index_list]
    path_list = []

    cnt = 0
    for path_idxs in path_index_lsit:
        path = []
        for i in range(len(path_idxs)):
            p = int(path_idxs[i])
            obj = {"value": values[p] , 
                "condition_str": "go to node {left} if X[:, {feature}] <= {threshold} else to node {right}.".format(
                    left=children_left[p],
                    feature=MUSIC_FEATURE[feature[p]],
                    threshold=threshold[p],
                    right=children_right[p],
                    value=values[p],
                ), 
                "condition": {"left":children_left[p],
                            "feature":MUSIC_FEATURE[feature[p]],
                            "threshold":threshold[p],
                            "right":children_right[p],
                            "value":values[p]},
                "depth":node_depth[p],
                "node":p,
                "is_leaf": is_leaves[p],
                "next": path_idxs[i+1] if i+1 < len(path_idxs) else -1,
                }
            path.append(obj)
        cnt += 1
        eval = get_evaluation(path, X, y, df)  
        info = {"path":path, "evaluation":eval}
        path_list.append(info)
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

    get_rules(forest.estimators_[0], X, y, df)


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
