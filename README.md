# XML-CNN

Implementation and enhancement of XML-CNN with respect to explainability and robustness of [Deep Learning for Extreme Multi-label Text Classification](http://nyc.lti.cs.cmu.edu/yiming/Publications/jliu-sigir17.pdf) using PyTorch.

Find the poster explanation video [here](https://drive.google.com/file/d/1y-tkqNtbWfCRvnc00jSRwmS2HJbVdQfd/view)

> Liu, J., Chang, W.-C., Wu, Y. and Yang, Y.: Deep learning fo extreme multi-label text classification, in Proc. of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 115-124 (2017).

# Requirements
- Python: 3.6.10 or higher
- PyTorch: 1.6.0 or higher
- Optuna: 2.0.0 or higher
- transformers
- lime
- shap

You can create a virtual environment of Anaconda from requirements.yml.  
We can't guarantee the operation of Anaconda environment other than the one created with requirements.yml.

```
$ conda env create -f requirements.yml
```

# Word Embeddings
Download GloVe embeddings from [Stanford's website](https://nlp.stanford.edu/projects/glove/):
```
$ mkdir -p .vector_cache
$ cd .vector_cache
$ wget http://nlp.stanford.edu/data/glove.6B.zip
$ unzip glove.6B.zip
$ cd ..
```

# Datasets
The dataset must be in the same format as it attached to this program.

It contains one document per line.  
It's stored in the order of ID, label, and text, separated by TAB from the left side.

```
{id}<TAB>{labels}<TAB>{texts}
```

## RCV1 Dataset
You can get the tokenized RCV1 dataset from [Lewis et al](https://www.jmlr.org/papers/volume5/lewis04a/lewis04a.pdf). by using this program's bundled get_rcv1.py.  
__Caution: This dataset is tokenized differently than the one used by Liu et al.__  
__Caution: If you want to use this dataset, please read the terms of use (Legal Issues) of the distribution destination.__

> Lewis, D. D.; Yang, Y.; Rose, T.; and Li, F. RCV1: A New Benchmark Collection for Text Categorization Research. Journal of Machine Learning Research, 5:361-397, 2004. http://www.jmlr.org/papers/volume5/lewis04a/lewis04a.pdf. 

> Lewis, D. D.  RCV1-v2/LYRL2004: The LYRL2004 Distribution of the RCV1-v2 Text Categorization Test Collection (12-Apr-2004 Version). http://www.jmlr.org/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm. 

## Eurlex-4K Dataset
You can also use the Eurlex-4K dataset for extreme multi-label classification:

1. Download the dataset from [XML repository](https://drive.google.com/file/d/0B3lPMIHmG6vGU0VTR1pCejFpWjg/view?resourcekey=0-SurjZ4z_5Tr38jENzf2Iwg)
2. Extract it inside the eurlex_data directory
3. Run preprocessing scripts in order:
   ```
   $ python eurlex_data/convert_eurlex.py eurlex_data/eurlex_train.txt data/train_org.txt
   $ python eurlex_data/convert_eurlex.py eurlex_data/eurlex_test.txt data/test.txt
   $ cd data
   $ python make_valid.py train_org.txt
   $ cd ..
   $ python eurlex_data/create_bow_embeddings.py --dim 5000 --emb-dim 300
   ```

# Dynamic Max Pooling
This program implements Dynamic Max Pooling based on the method by [Liu et al](http://nyc.lti.cs.cmu.edu/yiming/Publications/jliu-sigir17.pdf).

The p shown in that paper becomes the `d_max_pool_p` in params.yml.  
As in the paper, `d_max_pool_p` must be a divisible number for the output vector after convolution.

# Evaluation Metrics
Precision@K is available for this project.  
You can change it from params.yml.

# How to run
## When running at first

### Download GloVe Embeddings
Follow the instructions in the Word Embeddings section above.

### Download RCV1

Download datasets from http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm.  
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
$ python insert_labels.py  # Add appropriate labels to the dataset
```

### Directory Structure After Preprocessing
After successful preprocessing, your directory structure should look like:
```
data
   --train.txt
   --valid.txt
   --test.txt
eurlex_data
   --train.txt
   --valid.txt
   --test.txt
.vector_cache
   --glove.6B.300d.txt
   --bow_embeddings.txt
```

### Pre-trained Models
Pre-trained base and robust models trained on RCV1 and Eurlex-4K datasets can be downloaded from [here](https://drive.google.com/file/d/1bFFrAcyO6aBYujJOUBDMbXp7M2STS0kp/view).

Extract the zip file to create the .model_cache directory:
```
$ unzip XML-CNN-pre-trained-models.zip
```

## Training and Testing Options

### Normal Training
- model_name should be the name of dataset to be used (for ex. rcv1, eurlex)
- use -adv to load adv model
```
$ python train.py -m model_name
```

### Training Only
```
$ python train.py -m model_name -t
```

### Training with Adversarial Examples
```
$ python train.py -m model_name -t -adv
```

### Testing Only
```
$ python train.py -m model_name -te
```

### Testing with Adversarial Examples
```
$ python train.py -m model_name -te -adv
```

- **Running train.py will also run LIME and SHAP explanability on the trained model and save the results to the respective directory. Comment out specific lines in train.py to only run the explainability runs or only train or only test.**

## Params Search
```
$ python train.py --params_search
```
or
```
$ python train.py -s
```

## Force to use CPU
```
$ python train.py --use_cpu
```

# Contributors:

- [Vivek Sapkal](viveksapkal2003@gmail.com)
- [Preet Savalia](b22ai036@iitj.ac.in)

# Acknowledgment

This project is based on the following repositories.
Thank you very much for their accomplishments.

- [siddsax/XML-CNN](https://github.com/siddsax/XML-CNN) (MIT License)
- [PaulAlbert31/LabelNoiseCorrection](https://github.com/PaulAlbert31/LabelNoiseCorrection) (MIT License)