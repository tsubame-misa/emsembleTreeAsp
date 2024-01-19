import json
from api import getAllPastSongs
from analysis.randomForest import random_forest_classifier_maker
from analysis.logistic import logistic_classifier_maker
from analysis.svm import svm_classifier_maker

# 分類器のアップデート？をする


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

    random_forest_classifier_maker(formated_data)
    # logistic_classifier_maker(formated_data)
    # svm_classifier_maker(formated_data)

    print("fin make_classitier")


if __name__ == '__main__':
    main()
