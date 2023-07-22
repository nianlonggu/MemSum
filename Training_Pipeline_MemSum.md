<a href="https://colab.research.google.com/github/nianlonggu/MemSum/blob/main/Data_processing_training_and_testing_for_MemSum.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Preparation

## Clone the Repo


```python
!git clone https://github.com/nianlonggu/MemSum.git
```

    Cloning into 'MemSum'...
    remote: Enumerating objects: 203, done.[K
    remote: Counting objects: 100% (105/105), done.[K
    remote: Compressing objects: 100% (91/91), done.[K
    remote: Total 203 (delta 39), reused 54 (delta 9), pack-reused 98[K
    Receiving objects: 100% (203/203), 81.90 MiB | 16.89 MiB/s, done.
    Resolving deltas: 100% (58/58), done.


## Change the working directory to the main folder of MemSum


```python
import os
os.chdir("MemSum")
```

## Install packages

Note: Because colab has preinstalled torch, so we don't need to install pytorch again

We tested on torch version>=1.11.0.


```python
!pip install -r requirements.txt -q
```


```python
import torch
torch.__version__
```




    '1.12.1+cu113'



# Preprocessing Custom data

Suppose that you have already splitted the training / validation and  test set:

The training data is now stored in a .jsonl file that contains a list of json info, one line for one training instance. Each json (or dictonary) contains two keys: 

1. "text": the value for which is a python list of sentences, this represents the document you want to summarize;
2. "summary": the value is also a list of sentences. If represent the ground-truth summary. Because the summary can contain multiple sentences, so we store them as a list.

The same for the validation file and the testing file. 



```python
import json
train_corpus = [ json.loads(line) for line in open("data/custom_data/train_CUSTOM_raw.jsonl") ]

## as an example, we have 100 instances for training
print(len(train_corpus))
print(train_corpus[0].keys())
print(train_corpus[0]["text"][:3])
print(train_corpus[0]["summary"][:3])
```

    100
    dict_keys(['text', 'summary'])
    ['a recent systematic analysis showed that in 2011 , 314 ( 296 - 331 ) million children younger than 5 years were mildly , moderately or severely stunted and 258 ( 240 - 274 ) million were mildly , moderately or severely underweight in the developing countries .', 'in iran a study among 752 high school girls in sistan and baluchestan showed prevalence of 16.2% , 8.6% and 1.5% , for underweight , overweight and obesity , respectively .', 'the prevalence of malnutrition among elementary school aged children in tehran varied from 6% to 16% .']
    ['background : the present study was carried out to assess the effects of community nutrition intervention based on advocacy approach on malnutrition status among school - aged children in shiraz , iran.materials and methods : this case - control nutritional intervention has been done between 2008 and 2009 on 2897 primary and secondary school boys and girls ( 7 - 13 years old ) based on advocacy approach in shiraz , iran .', 'the project provided nutritious snacks in public schools over a 2-year period along with advocacy oriented actions in order to implement and promote nutritional intervention . for evaluation of effectiveness of the intervention growth monitoring indices of pre- and post - intervention were statistically compared.results:the frequency of subjects with body mass index lower than 5% decreased significantly after intervention among girls ( p = 0.02 ) .', 'however , there were no significant changes among boys or total population .']


If you have your own data, process them into the same structure then put them into the data/ folder

The next thing we need to do is to create high-ROUGE episodes for the training set, as introduced in the paper: https://aclanthology.org/2022.acl-long.450/,
and the github introduction: https://github.com/nianlonggu/MemSum#addition-info-code-for-obtaining-the-greedy-summary-of-a-document-and-creating-high-rouge-episodes-for-training-the-model


```python
from src.data_preprocessing.MemSum.utils import greedy_extract
import json
from tqdm import tqdm
```


```python
train_corpus = [ json.loads(line) for line in open("data/custom_data/train_CUSTOM_raw.jsonl") ]
for data in tqdm(train_corpus):
    high_rouge_episodes = greedy_extract( data["text"], data["summary"], beamsearch_size = 2 )
    indices_list = []
    score_list  = []

    for indices, score in high_rouge_episodes:
        indices_list.append( indices )
        score_list.append(score)

    data["indices"] = indices_list
    data["score"] = score_list
```

    100%|██████████| 100/100 [01:44<00:00,  1.05s/it]


Now we have obtained the labels for the training set. This can be parallized if you have large training set.

We can save the labeled training set to a new file:


```python
with open("data/custom_data/train_CUSTOM_labelled.jsonl","w") as f:
    for data in train_corpus:
        f.write(json.dumps(data) + "\n")
```

That's it! We are about to train MemSum!


```python

```

# Training

## Download pretrained word embedding

MemSUM used the glove embedding (200dim), with three addition token embeddings for bos eos pad, etc.

You can download the word embedding (a folder named glove/) used in this work:

https://drive.google.com/drive/folders/1lrwYrrM3h0-9fwWCOmpRkydvmF6hmvmW?usp=sharing

and put the folder under the model/ folder. 

Or you can do it using the code below:

Make sure the structure looks like:

1. MemSum/model/glove/unigram_embeddings_200dim.pkl
2. MemSum/model/glove/vocabulary_200dim.pkl


If not, you can change manually






```python
!pip install gdown -q
try:
    os.system("rm -r model")
    os.makedirs("model/")
except:
    pass
!cd model/; gdown --folder https://drive.google.com/drive/folders/1lrwYrrM3h0-9fwWCOmpRkydvmF6hmvmW


if not os.path.exists("model/glove"):
    try:
        os.makedirs("model/glove")
        os.system("mv model/*.pkl model/glove/")
    except:
        pass
```

    Retrieving folder list
    Processing file 1SVTHcgWJDvoVCsLfdvkaw5ICkihjUoaH unigram_embeddings_200dim.pkl
    Processing file 1SuF4HSe0-IBKWGtc1xqlzMHNDneiLi4- vocabulary_200dim.pkl
    Retrieving folder list completed
    Building directory structure
    Building directory structure completed
    Downloading...
    From: https://drive.google.com/uc?id=1SVTHcgWJDvoVCsLfdvkaw5ICkihjUoaH
    To: /content/MemSum/model/unigram_embeddings_200dim.pkl
    100% 320M/320M [00:01<00:00, 210MB/s]
    Downloading...
    From: https://drive.google.com/uc?id=1SuF4HSe0-IBKWGtc1xqlzMHNDneiLi4-
    To: /content/MemSum/model/vocabulary_200dim.pkl
    100% 4.16M/4.16M [00:00<00:00, 264MB/s]
    Download completed


## Start training

Note:
1. you need to switch to the folder src/MemSum_Full;
2. You can specify the path to training and validation set, the model_folder (where you want to store model checkpoints) and the log_folder (where you want to store the log info), and other parameters. 
3. You can provide the absolute path, or relative path, as shown in the example code below.
4. n_device means the number of available GPUs


```python
!cd src/MemSum_Full; python train.py -training_corpus_file_name ../../data/custom_data/train_CUSTOM_labelled.jsonl -validation_corpus_file_name ../../data/custom_data/val_CUSTOM_raw.jsonl -model_folder ../../model/MemSum_Full/custom_data/200dim/run0/ -log_folder ../../log/MemSum_Full/custom_data/200dim/run0/ -vocabulary_file_name ../../model/glove/vocabulary_200dim.pkl -pretrained_unigram_embeddings_file_name ../../model/glove/unigram_embeddings_200dim.pkl -max_seq_len 100 -max_doc_len 500 -num_of_epochs 10 -save_every 1000 -validate_every 1000 -n_device 1 -batch_size_per_device 4 -max_extracted_sentences_per_document 7 -moving_average_decay 0.999 -p_stop_thres 0.6
```

    100it [00:00, 10630.33it/s]
    100it [00:00, 13147.05it/s]
    /usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py:566: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      cpuset_checked))
    0it [00:00, ?it/s]train.py:227: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      remaining_mask_np = np.ones_like( doc_mask_np ).astype( np.bool ) | doc_mask_np
    train.py:228: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      extraction_mask_np = np.zeros_like( doc_mask_np ).astype( np.bool ) | doc_mask_np
    24it [00:34,  1.38s/it]Starting validation ...
    train.py:308: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      remaining_mask_np = np.ones_like( doc_mask ).astype( np.bool ) | doc_mask
    train.py:309: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      extraction_mask_np = np.zeros_like( doc_mask ).astype( np.bool ) | doc_mask
    val: 0.3103, 0.0874, 0.2774
    25it [00:48,  1.95s/it]
    24it [00:37,  1.63s/it]Starting validation ...
    val: 0.3175, 0.0926, 0.2828
    25it [00:53,  2.12s/it]
    24it [00:37,  1.54s/it]Starting validation ...
    val: 0.3223, 0.0957, 0.2860
    25it [00:53,  2.13s/it]
    24it [00:38,  1.62s/it][current_batch: 00100] loss: 0.485, learning rate: 0.000100
    Starting validation ...
    val: 0.3287, 0.1040, 0.2935
    25it [00:54,  2.17s/it]
    24it [00:38,  1.57s/it]Starting validation ...
    val: 0.3359, 0.1081, 0.3000
    25it [00:53,  2.13s/it]
    24it [00:38,  1.62s/it]Starting validation ...
    val: 0.3427, 0.1117, 0.3064
    25it [00:53,  2.16s/it]
    24it [00:38,  1.60s/it]Starting validation ...
    val: 0.3472, 0.1140, 0.3107
    25it [00:54,  2.17s/it]
    24it [00:38,  1.61s/it][current_batch: 00200] loss: 0.469, learning rate: 0.000100
    Starting validation ...
    val: 0.3524, 0.1136, 0.3137
    25it [00:55,  2.20s/it]
    24it [00:38,  1.61s/it]Starting validation ...
    val: 0.3603, 0.1184, 0.3216
    25it [00:53,  2.15s/it]
    24it [00:39,  1.66s/it]Starting validation ...
    val: 0.3635, 0.1206, 0.3258
    25it [00:54,  2.19s/it]


# Testing trained model on custom dataset


```python
from summarizers import MemSum
from tqdm import tqdm
from rouge_score import rouge_scorer
import json
import numpy as np
```


```python
rouge_cal = rouge_scorer.RougeScorer(['rouge1','rouge2', 'rougeLsum'], use_stemmer=True)

memsum_custom_data = MemSum(  "model/MemSum_Full/custom_data/200dim/run0/model_batch_250.pt", 
                  "model/glove/vocabulary_200dim.pkl", 
                  gpu = 0 ,  max_doc_len = 500  )
```


```python
test_corpus_custom_data = [ json.loads(line) for line in open("data/custom_data/test_CUSTOM_raw.jsonl")]
```


```python
def evaluate( model, corpus, p_stop, max_extracted_sentences, rouge_cal ):
    scores = []
    for data in tqdm(corpus):
        gold_summary = data["summary"]
        extracted_summary = model.extract( [data["text"]], p_stop_thres = p_stop, max_extracted_sentences_per_document = max_extracted_sentences )[0]
        
        score = rouge_cal.score( "\n".join( gold_summary ), "\n".join(extracted_summary)  )
        scores.append( [score["rouge1"].fmeasure, score["rouge2"].fmeasure, score["rougeLsum"].fmeasure ] )
    
    return np.asarray(scores).mean(axis = 0)
```


```python
evaluate( memsum_custom_data, test_corpus_custom_data, 0.6, 7, rouge_cal )
```

    100%|██████████| 100/100 [00:16<00:00,  5.90it/s]





    array([0.37957819, 0.13561023, 0.3435555 ])




```python

```

To cite MemSum, please use the following bibtex:

```
@inproceedings{gu-etal-2022-memsum,
    title = "{M}em{S}um: Extractive Summarization of Long Documents Using Multi-Step Episodic {M}arkov Decision Processes",
    author = "Gu, Nianlong  and
      Ash, Elliott  and
      Hahnloser, Richard",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.450",
    doi = "10.18653/v1/2022.acl-long.450",
    pages = "6507--6522",
    abstract = "We introduce MemSum (Multi-step Episodic Markov decision process extractive SUMmarizer), a reinforcement-learning-based extractive summarizer enriched at each step with information on the current extraction history. When MemSum iteratively selects sentences into the summary, it considers a broad information set that would intuitively also be used by humans in this task: 1) the text content of the sentence, 2) the global text context of the rest of the document, and 3) the extraction history consisting of the set of sentences that have already been extracted. With a lightweight architecture, MemSum obtains state-of-the-art test-set performance (ROUGE) in summarizing long documents taken from PubMed, arXiv, and GovReport. Ablation studies demonstrate the importance of local, global, and history information. A human evaluation confirms the high quality and low redundancy of the generated summaries, stemming from MemSum{'}s awareness of extraction history.",
}
```


```python

```
