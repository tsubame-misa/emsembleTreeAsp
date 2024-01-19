import re
import json
from api import getAllPastSongs
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np


MUSIC_FEATURE = ["tempo", "danceability", "energy", "mode", "loudness", "acousticness", "speechiness", "instrumentalness",
                 "liveness", "key", "valence", "duration_ms", "time_signature"]

def ranking_convert(x):
    if int(x) <= 20:
        return 1
    else:
        return 0

# TODO:このデータの整形をなしの形でapiからとってきたい
def formatData(data):
    music_feature_key = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                         'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']
    s_data = []
    for d in data:
        obj = dict()
        obj["rank"] = d["rank"]
        if d["detail"].get("music_feature"):
            for key in music_feature_key:
                # print(key)
                # print(type(d["detail"]["music_feature"]))
                if type(d["detail"]["music_feature"]) is str:
                    d["detail"]["music_feature"] = json.loads(d["detail"]["music_feature"])
                obj[key] = d["detail"]["music_feature"][key]
        else:
            continue
        s_data.append(obj)
    return s_data


def main():
    data = getAllPastSongs()
    formated_data = formatData(data)

    input_string = "selected(0) selected(53) selected(54) selected(96) selected(383) selected(426) selected(488) selected(503) selected(618) selected(779) selected(19) selected(22) selected(39) selected(506) selected(620) selected(636) selected(696) selected(838) selected(839) selected(843)"
    # 正規表現を使用して数字のみを抽出
    numbers = [int(match.group()) for match in re.finditer(r'\d+', input_string)]

    # ruleIdと条件が紐づいた辞書の読み込み
    with open('./classifier/rule_condition-5.json') as f:
        conditions = json.load(f)
    
    # testデータのindexを取得
    with open('./classifier/train-test-index4.json') as f:
        obj = json.load(f)
        test_index = np.array(obj["test"])
    
    # モデルのオープン
    with open("./classifier/analysis/models/randomForestModel-4.pickle", mode='rb') as f:
        forest = pickle.load(f)

    df = pd.DataFrame(formated_data)
    # ランキングを01で
    df["rank"] = df["rank"].apply(ranking_convert)
    
    # X = df.loc[:, MUSIC_FEATURE].values
    y = df["rank"]

    # test_X, test_y =  X[test_index], y[test_index]
    # dt_test = df.loc[test_index, :]

    dt_test = df.loc[test_index, :]
    X = dt_test.loc[:, MUSIC_FEATURE].values
    test_y = dt_test["rank"]
    predict_y = []
    can_predict = 0

    predict_y = []
    can_predict = 0
    forest_score = []

    X_idx = 0
    for index, row in dt_test.iterrows():
        for rule_num in numbers:
            found = True
            cs = conditions[str(rule_num)]["condition"]
            for c in cs:
                if c["leq"]:
                    if not row[c["feature"]] <= c["threshold"]:
                        found = False
                        break
                else:
                    if not row[c["feature"]] > c["threshold"]:
                        found = False
                        break    
            if found:
                can_predict += 1
                print(conditions[str(rule_num)]["class"], y[X_idx] )
                predict_y.append(conditions[str(rule_num)]["class"])
                break
        if found==False:
            res = forest.predict([X[X_idx]])
            predict_y.append(res[0])
            forest_score.append(res[0])
        X_idx+=1
    
    print(len(predict_y), len(test_y))
    print("can",can_predict)

    print("ASP-----------------------")
    print('accuracy = ', accuracy_score(y_true=test_y, y_pred=predict_y))
    print('precision = ', precision_score(y_true=test_y, y_pred=predict_y))
    print('recall = ', recall_score(y_true=test_y, y_pred=predict_y))
    print('f1 score = ', f1_score(y_true=test_y, y_pred=predict_y))
    print('confusion matrix = \n', confusion_matrix(y_true=test_y, y_pred=predict_y))

    print("fin make_classitier")


if __name__ == '__main__':
    main()
