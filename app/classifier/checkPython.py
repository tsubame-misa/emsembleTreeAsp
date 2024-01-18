import re
import json
from api import getAllPastSongs
from analysis.randomForest import random_forest_classifier_maker
from analysis.logistic import logistic_classifier_maker
from analysis.svm import svm_classifier_maker
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


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
        # THINK:歌詞データがないのを省くでいいのかどうか（現状は省いている）
        # if d["detail"].get("lyrics_feature"):
        #     if d["detail"]["lyrics_feature"]["total_rhyme_score"] is None:
        #         continue
        #     else:
        #         obj["total_rhyme_score"] = d["detail"]["lyrics_feature"]["total_rhyme_score"]
        #     if d["detail"]["lyrics_feature"]["total_positive_score"] is None:
        #         continue
        #     else:
        #         obj["total_positive_score"] = d["detail"]["lyrics_feature"]["total_positive_score"]
        # else:
        #     continue
        s_data.append(obj)
    return s_data


def main():
    data = getAllPastSongs()
    formated_data = formatData(data)

    input_string = "selected(118) selected(181) selected(92) selected(95) selected(182) selected(216) selected(221) selected(390) selected(393) selected(495)"
    # 正規表現を使用して数字のみを抽出
    numbers = [int(match.group()) for match in re.finditer(r'\d+', input_string)]

    # ruleIdと条件が紐づいた辞書の読み込み
    with open('./classifier/rule_condition.json') as f:
        conditions = json.load(f)
    

    # モデルのオープン
    with open("./classifier/analysis/models/randomForestModel.pickle", mode='rb') as f:
        forest = pickle.load(f)


    df = pd.DataFrame(formated_data)
    # ランキングを01で
    df["rank"] = df["rank"].apply(ranking_convert)

    df_train, df_test = train_test_split(df, random_state=123)

    X = df_test.loc[:, MUSIC_FEATURE].values
    y = df_test["rank"]
    predict_y = []
    can_predict = 0

    forest_score = []

    X_idx = 0
    for index, row in df_test.iterrows():
        for rule_num in numbers:
            found = True
            # print(conditions[str(rule_num)])
            cs = conditions[str(rule_num)]["condition"]
            for c in cs:
                # print(c)
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
                print(conditions[str(rule_num)]["class"])
                predict_y.append(conditions[str(rule_num)]["class"])
                break
        res = forest.predict([X[X_idx]])
        if found==False:
            res = forest.predict([X[X_idx]])
            predict_y.append(res[0])
        forest_score.append(res[0])
        X_idx+=1
    
    print(len(predict_y), len(y))
    print("can",can_predict)

    print("ASP-----------------------")
    print('accuracy = ', accuracy_score(y_true=y, y_pred=predict_y))
    print('precision = ', precision_score(y_true=y, y_pred=predict_y))
    print('recall = ', recall_score(y_true=y, y_pred=predict_y))
    print('f1 score = ', f1_score(y_true=y, y_pred=predict_y))
    print('confusion matrix = \n', confusion_matrix(y_true=y, y_pred=predict_y))

    print("original-----------------------")
    print('accuracy = ', accuracy_score(y_true=y, y_pred=forest_score))
    print('precision = ', precision_score(y_true=y, y_pred=forest_score))
    print('recall = ', recall_score(y_true=y, y_pred=forest_score))
    print('f1 score = ', f1_score(y_true=y, y_pred=forest_score))
    print('confusion matrix = \n', confusion_matrix(y_true=y, y_pred=forest_score))


    print("fin make_classitier")


if __name__ == '__main__':
    main()
