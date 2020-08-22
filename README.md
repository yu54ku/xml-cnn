[日本語](https://github.com/yu54ku/xml-cnn/blob/master/README_J.md)

# XML-CNN
Implementation of [Deep Learning for Extreme Multi-label Text Classification](http://nyc.lti.cs.cmu.edu/yiming/Publications/jliu-sigir17.pdf) using PyTorch.

# System Requirements
- Python: 3.6.10 or higher
- PyTorch: 1.6.0 or higher
- torchtext: 0.6.0 or higher
- Gensim: 3.8.0 or higher
- Optuna: 2.0.0 or higher

You can create a virtual environment of Anaconda from `requirements.yml`.

```
$ conda env create -f requirements.yml
```


# Datasets
The dataset must be in the same format as it attached to this program.

It contains one document per line.
It's stored in the order of ID, label, and text, separated by TAB from the left side.

```
{id}<TAB>{labels}<TAB>{texts}
```

You can get the pre-processed RCV1 dataset from Lewis et al. by using this program's bundled `data/get_rcv1.py`.


# Dynamic Max Pooling
This program implements Dynamic Max Pooling based on the method by [Liu et al](http://nyc.lti.cs.cmu.edu/yiming/Publications/jliu-sigir17.pdf).

The p shown in that paper becomes the `"d_max_pool_p"` in `./params.yml`.

As in the paper, `"d_max_pool_p"` must be a divisible number for the output vector after convolution.


# Evaluation Metrics
Precision@K and F1-Score are available for this program.

You can change it from `./params.yml`.

# How to run
## When first run
### Donwload RCV1

Donwload datasets from http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm

```
$ cd data
$ python get_rcv1.py
```
> Lewis, D. D.; Yang, Y.; Rose, T.; and Li, F. RCV1: A New Benchmark Collection for Text Categorization Research. Journal of Machine Learning Research, 5:361-397, 2004. http://www.jmlr.org/papers/volume5/lewis04a/lewis04a.pdf. 


### Make valid dataset

```
$ python make_valid.py train_org.txt
```

### Run

```
$ python train.py
```

## Normal Training

```
$ python train.py
```

## Params Search
```
$ python train.py --params_search
```
or
```
$ python train.py -s
```
## Force to use cpu

```
$ python train.py --use_cpu
```

# Acknowledgment
This program is based on the following repositories.
Thank you very much for their accomplishments.


- [siddsax/XML-CNN](https://github.com/siddsax/XML-CNN) (MIT license)
- [PaulAlbert31/LabelNoiseCorrection](https://github.com/PaulAlbert31/LabelNoiseCorrection) (MIT License)
