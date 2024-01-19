以下を環境変数に設定してください

```
SPOTIFY_CLIENT_ID= 'hoge'
SPOTIFY_CLIENT_SECRET_ID= 'hoge'
MUSIXMATCH_API_KEY = "hoge"
HASURA_ADMIN_SECRET="hoge"
```

ルートディレクトリ直下で`python classifier/classifierMaker.py`で分類器の生成・アップデートができます。

`data`フォルダに spotify chart からダウンロードした csv ファイルを置き、`python maker.py`ファイルのパスを書き換え実行するとデータを抽出・分類しバズると予測された曲をデータベースに追加します。

本レポートでは rule ディレクトリ, classifier ディレクトリ以下のみの使用です
