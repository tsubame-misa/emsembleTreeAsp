from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import json
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold

from .common import MUSIC_FEATURE
from .formated import format_data

import os

file_path = os.path.dirname(os.path.realpath(__file__))


global coundition_count
global rule_id
global conditions 
global rule_id_conditions_dict

def convert_rule(info):
    global rule_id, coundition_count, conditions, rule_id_conditions_dict
    idx = rule_id
    eval = info["evaluation"]
    path_info = info["path_info"]
    support = 3 #全体の長さ
    size = 0 #分割条件の数

    rule = "rule("+str(idx)+"). "
    
    if not eval["predict_class"] is None:
        rule += "accuracy({idx},{accuracy}). error_rate({idx},{error_rate}). precision({idx},{precision}). "\
            "recall({idx},{recall}). f1_score({idx},{f1}). predict_class({idx},{predict_class}). ".format(
            idx=idx,
            accuracy = eval["accuracy"],
            error_rate = eval["error_rate"],
            precision = eval["precision"],
            recall = eval["recall"],
            f1 = eval["f1"],
            predict_class = eval["predict_class"],
            )
        support += 6
    
    rule_condition = []

    for i in range(len(path_info)):
        rule += "condition({idx}, {coundition_count}). ".format(idx=idx, coundition_count=coundition_count)
        
        # conditionの中身は見ていないので、辞書で対応づけられる数値を入れておく
        condition = path_info[i]["condition"]
        c = {"feature": condition["feature"], 
            "threshold": float(condition["threshold"]),
            "leq": bool(condition["left"] ==  path_info[i]["next"]),
            }
        conditions.append(c)
        rule_condition.append(c)
        coundition_count += 1
        support += 1
        size += 1
    
    rule_id_conditions_dict[idx] = {"condition":rule_condition, "class": 0 if condition["value"][0][0] > condition["value"][0][1] else 1}
    
    rule += "support({idx},{support}). size({idx},{size}).".format(idx=idx, support=support, size=size)   

    rule_id += 1

    return rule 



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
            if i==len(path)-1:
                condition = path[-1]["condition"]
            if condition["value"][0][0] > condition["value"][0][1]:
                predict_class = 0
                predict_y.append(0)
            else:
                predict_class = 1
                predict_y.append(1)
            ans_y.append(y[index])
            predicted_x_index.append(index)

    if len(predict_y) > 0:
        # 小数点が扱えないので100%表記に
        eval = {"accuracy": int(accuracy_score(y_true=ans_y, y_pred=predict_y)*100),
                "precision": int(precision_score(y_true=ans_y, y_pred=predict_y)*100),
                "recall": int(recall_score(y_true=ans_y, y_pred=predict_y)*100),
                "f1": int(f1_score(y_true=ans_y, y_pred=predict_y)*100), 
                "error_rate" : int((1 - accuracy_score(y_true=ans_y, y_pred=predict_y))*100),
                "predict_class": predict_class}
        return eval

    return {"accuracy":None,"precision":None,"recall": None,"f1":None, "error_rate":None, "predict_class":None}




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
    for j in range(len(path_index_lsit)):
        path_idxs = path_index_lsit[j]
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
        eval = get_evaluation(path, X, y, df)
        # そのパスを通るデータがない時
        if eval["predict_class"] is None:
            continue
        info = {"path":path_idxs, "path_info":path, "evaluation":eval}
        rule = convert_rule(info)
        info["rule"] = rule

        path_list.append(info)
    print(len(path_list))
    return path_list
    

# RandomForestの分類器を作る
def random_forest_classifier_maker(data):
    
    df = format_data(data)
    df_train, df_test = train_test_split(df)

    X = df_train.loc[:, MUSIC_FEATURE].values
    y = df_train["rank"]

    # skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    # for train_index, test_index in skf.split(X, y):
    #     print("train_index:", len(train_index), "test_index:", len(test_index))
    #     exit()


    # ランダムフォレスト回帰
    forest = RandomForestClassifier(random_state=1234, n_estimators=5)
    # forest = RandomForestClassifier(random_state=1234, n_estimators=5, max_depth=10, max_leaf_nodes=30)
    # モデル学習
    forest.fit(X, y)

    # 学習モデルの保存
    with open(file_path+'/models/randomForestModel.pickle', mode='wb') as f:
        pickle.dump(forest, f, protocol=2)

    y_train_pred = forest.predict(X)
    print('accuracy = ', accuracy_score(y_true=y, y_pred=y_train_pred))
    print('precision = ', precision_score(y_true=y, y_pred=y_train_pred))
    print('recall = ', recall_score(y_true=y, y_pred=y_train_pred))
    print('f1 score = ', f1_score(y_true=y, y_pred=y_train_pred))
    print('confusion matrix = \n', confusion_matrix(y_true=y, y_pred=y_train_pred))
    print("-------------------")
 
    global coundition_count, conditions, rule_id, rule_id_conditions_dict
    coundition_count = 0
    conditions = []
    rule_id = 0
    rule_id_conditions_dict = {}

    rules = []
    all_info = []
    support_value = []

    print(len(forest.estimators_))
    cnt = 0
    for tree in forest.estimators_:
        print("#################################")
        print("tree number",cnt)
        print("#################################")
        cnt+=1
        tree_rules = get_rules(tree, X, y, df_train)

        for t in tree_rules:
            rules.append(t["rule"])
            support_value.append(len(t["path"]))
            all_info.append(t)

    print("support avarage", sum(support_value)/len(support_value))
    
    # ルールの出力
    rule_str = '\n'.join(rules)
    f = open('rules.txt', 'w')
    f.write(rule_str)
    f.close()

    with open('condition.json', 'w') as f:
        json.dump(conditions, f)

    with open('rule_condition.json', 'w') as f:
        json.dump(rule_id_conditions_dict, f)
    

    X = df_test.loc[:, MUSIC_FEATURE].values
    y = df_test["rank"]
    y_train_pred = forest.predict(X)
    print('accuracy = ', accuracy_score(y_true=y, y_pred=y_train_pred))
    print('precision = ', precision_score(y_true=y, y_pred=y_train_pred))
    print('recall = ', recall_score(y_true=y, y_pred=y_train_pred))
    print('f1 score = ', f1_score(y_true=y, y_pred=y_train_pred))
    print('confusion matrix = \n', confusion_matrix(y_true=y, y_pred=y_train_pred))



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
