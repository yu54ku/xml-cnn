[日本語](https://github.com/yu54ku/xml-cnn/blob/master/README_J.md)

# XML-CNN
Implementation of [Deep Learning for Extreme Multi-label Text Classification](http://nyc.lti.cs.cmu.edu/yiming/Publications/jliu-sigir17.pdf) using PyTorch.

> Liu, J., Chang, W.-C., Wu, Y. and Yang, Y.: Deep learning fo extreme multi-label text classification, in Proc. of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 115-124 (2017).

# Requirements
- Python: 3.6.10 or higher
- PyTorch: 1.6.0 or higher
- torchtext: 0.6.0 or higher
- Optuna: 2.0.0 or higher

You can create a virtual environment of Anaconda from `requirements.yml`.  
We can't guarantee the operation of Anaconda environment other than the one created with `requirements.yml`.

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

You can get the tokenized RCV1 dataset from [Lewis et al](https://www.jmlr.org/papers/volume5/lewis04a/lewis04a.pdf). by using this program's bundled `data/get_rcv1.py`.  
__Caution: This dataset is tokenized differently than the one used by Liu et al.__  
__Caution: If you want to use this dataset, please read the terms of use (Legal Issues) of the distribution destination.__

> Lewis, D. D.; Yang, Y.; Rose, T.; and Li, F. RCV1: A New Benchmark Collection for Text Categorization Research. Journal of Machine Learning Research, 5:361-397, 2004. http://www.jmlr.org/papers/volume5/lewis04a/lewis04a.pdf. 

> Lewis, D. D.  RCV1-v2/LYRL2004: The LYRL2004 Distribution of the RCV1-v2 Text Categorization Test Collection (12-Apr-2004 Version). http://www.jmlr.org/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm. 


# Dynamic Max Pooling
This program implements Dynamic Max Pooling based on the method by [Liu et al](http://nyc.lti.cs.cmu.edu/yiming/Publications/jliu-sigir17.pdf).

## Known issue: difference from the paper
In the paper (Section 3.1), the feature map of **each filter** is divided into p chunks, so the pooling layer outputs `filter_channels × p` features for each filter size.
In this implementation, the feature maps of all filters are concatenated into one sequence before pooling, so `d_max_pool_p` is the **total number of chunks across all filters** of each filter size, not the paper's p.

- Setting `d_max_pool_p = filter_channels × p` (where the length of the convolution output is divisible by p) gives exactly the same pooling as the paper.
- Other values, including the defaults in `./params.yml`, create pooling windows that span multiple filters. This is not described in the paper. With the defaults, filter sizes 4 and 8 behave like standard max-over-time pooling (p = 1).
- The parameter search also chooses from divisors of `filter_channels × (length of the convolution output)`, so most candidates differ from the paper's pooling.

I intend to fix this issue, but I cannot promise when.
Until then, if you need an implementation that follows the paper, please consider
[LibMultiLabel](https://github.com/ASUS-AICS/LibMultiLabel).


# Evaluation Metrics
Precision@K and F1-Score are available for this program.  
You can change it from `./params.yml`.

# How to run
## When running at first
### Donwload RCV1

Donwload datasets from http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm.  
__Caution: This dataset is tokenized differently than the one used by Liu et al.__  
__Caution: If you want to use this dataset, please read the terms of use (Legal Issues) of the distribution destination.__

```
$ cd data
$ python get_rcv1.py
```


### Make valid dataset

```
$ python make_valid.py train_org.txt
```


### Make dataset for Params Search

```
$ python make4search.py train_org.txt
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


- [siddsax/XML-CNN](https://github.com/siddsax/XML-CNN) (MIT License)
- [PaulAlbert31/LabelNoiseCorrection](https://github.com/PaulAlbert31/LabelNoiseCorrection) (MIT License)
