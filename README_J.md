[English](https://github.com/yu54ku/xml-cnn/blob/master/README.md)

# XML-CNN
PyTorchを用いた [Deep Learning for Extreme Multi-label Text Classification](http://nyc.lti.cs.cmu.edu/yiming/Publications/jliu-sigir17.pdf) の実装．

> Liu, J., Chang, W.-C., Wu, Y. and Yang, Y.: Deep learning fo extreme multi-label text classification, in Proc. of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 115-124 (2017).

# 動作環境
- Python: 3.6.10 以上
- PyTorch: 1.6.0 以上
- torchtext: 0.6.0 以上
- Optuna: 2.0.0 以上

`requirements.yml` を用いることで，動作確認済みの仮想環境をAnacondaで作成することが出来ます．  
`requirements.yml` を用いて作成されたAnaconda環境以外での動作は保証出来ません．

```
$ conda env create -f requirements.yml
```


# データセット
付属のデータセットと同じ形式で入力してください．

1行に1文書が対応しています．  
左から順にID，ラベル，テキストの順で，TAB区切りになっています．

```
{id}<TAB>{labels}<TAB>{texts}
```
このプログラムに含まれる `data/get_rcv1.py` を用いることで，[Lewisら](https://www.jmlr.org/papers/volume5/lewis04a/lewis04a.pdf) によって前処理された RCV1 データセットをダウンロードすることが出来ます．  
__注意: Liuらの手法で使用されているデータセットとは前処理が異なります.__  
__注意: このデータセットを使用する際には，配布元の利用規約(Regal Issues)を参照してください．__

> Lewis, D. D.; Yang, Y.; Rose, T.; and Li, F. RCV1: A New Benchmark Collection for Text Categorization Research. Journal of Machine Learning Research, 5:361-397, 2004. http://www.jmlr.org/papers/volume5/lewis04a/lewis04a.pdf. 

> Lewis, D. D. RCV1-v2/LYRL2004: The LYRL2004 Distribution of the RCV1-v2 Text Categorization Test Collection (12-Apr-2004 Version). http://www.jmlr.org/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm.

# Dynamic Max Pooling
このプログラムでは，[Liuらの手法](http://nyc.lti.cs.cmu.edu/yiming/Publications/jliu-sigir17.pdf) を基にDynamic Max Poolingを実装しています．

## 既知の問題：論文との違い
論文（Section 3.1）では，**フィルタごとに**特徴マップを p 個のチャンクに分割するため，プーリング層の出力はフィルタサイズごとに `filter_channels × p` 次元になります．
このプログラムでは，全フィルタの特徴マップを1本につなげてから分割しているため，`d_max_pool_p` は論文の p ではなく，**フィルタサイズごとの全フィルタ合計のチャンク数**になっています．

- `d_max_pool_p = filter_channels × p`（ただし畳み込み後の長さが p で割り切れる場合）とすると，論文と完全に同じプーリングになります．
- それ以外の値（`./params.yml` の既定値を含む）では，複数のフィルタにまたがる窓でmaxをとることになり，これは論文にない操作です．既定値では，フィルタサイズ4と8は通常のmax-over-time pooling（p = 1）と同じ動作になります．
- パラメータサーチも `filter_channels × 畳み込み後の長さ` の約数から選んでいるため，候補の多くは論文のプーリングと異なります．

修正を予定しています．論文の結果と比較する際はご注意ください．

# 評価尺度
このプログラムには Precision@K と F1-Score が用意されています．  
`./params.yml` から変更が可能です．

# 実行方法
このプログラムには `--params_search` と `--use_cpu` オプションがあります．

## 初回実行時
### RCV1のダウンロード

http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm から前処理済みの RCV1 データセットをダウンロードします．  
__注意: Liuらの手法で使用されているデータセットとは前処理が異なります.__  
__注意: このデータセットを使用する際には，配布元の利用規約(Regal Issues)を参照してください．__

```
$ cd data
$ python get_rcv1.py
```

### 検証用データの生成

```
$ python make_valid.py train_org.txt
```

### パラメータサーチ用データの生成

```
$ python make4search.py train_org.txt
```

### 実行

```
$ python train.py
```
## 通常の学習

```
$ python train.py
```

## パラメータサーチ

```
$ python train.py --params_search
```
もしくは
```
$ python train.py -s
```

## 強制的にCPUを使用

```
$ python train.py --use_cpu
```

# 謝意
このプログラムは以下のリポジトリを基にしています．彼らの成果に感謝します.


- [siddsax/XML-CNN](https://github.com/siddsax/XML-CNN) (MIT License)
- [PaulAlbert31/LabelNoiseCorrection](https://github.com/PaulAlbert31/LabelNoiseCorrection) (MIT License)
