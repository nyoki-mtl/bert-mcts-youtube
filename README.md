# BERT-MCTS-YOUTUBE

YouTubeにてヨビノリたくみさんと対戦した将棋ソフトです。
自然言語モデルであるBERTとモンテカルロ木探索(MCTS)の組み合わせで出来ています。  
すべてpythonで書いてあるため、探索の速度は遅いです。  

BERT以外の大部分は『将棋AIで学ぶディープラーニング』を参考に書いています。
- [書籍(amazon)](https://www.amazon.co.jp/dp/B07B7JJ929)
- [github](https://github.com/TadaoYamaoka/python-dlshogi)

## 環境

### Colab

テストするだけなら[google colab](https://colab.research.google.com/drive/10KAuLlNe6FKZBp_iE2bQJPNhoY2WeACx?usp=sharing) が簡単です。

以下はローカルで試す場合。CPUだと遅いのでCUDA環境が望ましいです。  
重みファイルは[ここ](https://drive.google.com/drive/folders/1N-Np2NmNLtLGS9gjnreBkYdTxrH1EHFw?usp=sharing) にアップしてあり、
たくみさんと戦った重みファイルがyoutube_version.ckpt、追加で数日間学習させた重みファイルがlatest.ckptになります。  
ダウンロード先のパスはengine/***_player.sh内で指定してください。
デフォルトではwork_dirs以下にダウンロードすることを想定しています。

### Docker

cuda10.2以上のnvidia-dockerが整っているなら次のコマンドで環境に入れます。
```bash
$ make docker-start-interactive
```

### Ubuntu18.04

cuda10.2でanacondaが入っていれば次のコマンドで仮想環境を作れます。
```bash
$ conda env create -f env_name.yml
$ conda activate bert-mcts-youtube
$ python setup.py develop
```

### Windows10

未検証

## 将棋エンジンのテスト

エンジンはengineディレクトリに用意しています。これらはShogiGUIなどから呼び出すことができます。
- policy_player.shはBERTの出力する方策のみを頼りに指すモデル(弱い)
- mcts_player.shはBERTの出力をもとにMCTSで探索するモデル

## 学習

学習には互角局面集とGCTの自己対戦棋譜を用いました。  
モデルはMasked Language Modelで事前学習してから、Policy Value Networkの学習という手順を踏みます。  
ただし、将棋は良質な教師データが大量にあるため事前学習の効果はあまりない気がします。

### データの準備

互角局面集のダウンロード
```bash
$ cd data
$ git clone https://github.com/tttak/ShogiGokakuKyokumen.git
```

GCTの自己対戦棋譜
```bash
$ cd data
$ mkdir hcpe
```

GCTの自己対戦棋譜は開発者の加納さんが[リンク](https://drive.google.com/drive/folders/14FaqqIHRctTQIY6hScCFXWQQZ_pSU3-F)
に公開してくださっていまし。
ここからselfplay-***となっているファイルをいくつかdata/hcpe以下にダウンロードしてください。  
サイズが大きいので一個でも十分な量あります。

これらを準備できたら以下のコマンドでデータセットを作ります。

```bash
$ python tools/make_dataset.py
```

### Masked Language Modelの学習

```bash
$ python tools/train.py configs/mlm_base.yaml
```

### 重みファイルの変換

Masked Language Modelのチェックポイントをtransformers形式に変換しておきます。
これによって転移学習のコードが多少書きやすくなります。

```bash
$ python tools/pl_to_transformers.py work_dirs/mlm_base/version_0/checkpoints/last.ckpt
```

### Policy Value Modelの学習

最後にこれらを使ってPolicy Valueを学習させます。

```bash
$ python tools/train.py configs/policy_value.yaml
```