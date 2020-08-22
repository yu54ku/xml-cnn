[English](https://github.com/yu54ku/xml-cnn/blob/master/README.md)

# XML-CNN
PyTorchを用いた [Deep Learning for Extreme Multi-label Text Classification](http://nyc.lti.cs.cmu.edu/yiming/Publications/jliu-sigir17.pdf) の実装．

> Liu, J., Chang, W.-C., Wu, Y. and Yang, Y.: Deep learning fo extreme multi-label text classification, in Proc. of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 115 - 124 (2017).

# 動作環境
- Python: 3.6.10 以上
- PyTorch: 1.6.0 以上
- torchtext: 0.6.0 以上
- Optuna: 2.0.0 以上

`./requirements.yml` を用いることで，動作確認済みの仮想環境をAnacondaで作成することが出来ます．

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


# Dynamic Max Pooling
このプログラムでは，[Liuらの手法](http://nyc.lti.cs.cmu.edu/yiming/Publications/jliu-sigir17.pdf) を基にDynamic Max Poolingを実装しています．

論文における p は `./params.yml` の `"d_max_pool_p"` となります．

`"d_max_pool_p"` は論文と同様，畳み込み後の出力ベクトルに対して割り切れる数でなければなりません．手動でパラメータを設定する場合は注意してください．

このプログラムのパラメータサーチでは，割り切れる数を列挙しその中からパラメータを選択しています．

# 評価尺度
このプログラムには Precision@K と F1-Score が用意されています．

`./params.yml` から変更が可能です．

# 実行方法
このプログラムには `--params_search` と `--use_cpu` オプションがあります．

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


- [siddsax/XML-CNN](https://github.com/siddsax/XML-CNN) (MIT license)
- [PaulAlbert31/LabelNoiseCorrection](https://github.com/PaulAlbert31/LabelNoiseCorrection) (MIT License)
