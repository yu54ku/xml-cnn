[日本語](https://github.com/yu54ku/xml-cnn/blob/master/README_J.md)

# XML-CNN
Implementation of [Deep Learning for Extreme Multi-label Text Classification](http://nyc.lti.cs.cmu.edu/yiming/Publications/jliu-sigir17.pdf) using Pytorch.

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

# Dynamic Max Pooling
This program implements Dynamic Max Pooling based on the method by [Liu et al](http://nyc.lti.cs.cmu.edu/yiming/Publications/jliu-sigir17.pdf).

The p shown in that paper becomes the `"d_max_pool_p"` in `./params.yml`.

As in the paper, `"d_max_pool_p"` must be a divisible number for the output vector after convolution.


# Evaluation Metrics
Precision@K and F1-Score are available for this program.

You can change it from `./params.yml`.

# How to run
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
## Forced use of CPU

```
$ python train.py --use_cpu
```

# Acknowledgment
This program is based on the following repositories.
Thank you very much for their accomplishments.


- [siddsax/XML-CNN](https://github.com/siddsax/XML-CNN) (MIT license)
- [PaulAlbert31/LabelNoiseCorrection](https://github.com/PaulAlbert31/LabelNoiseCorrection) (MIT License)
