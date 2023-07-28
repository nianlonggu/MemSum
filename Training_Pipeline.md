# Preparation

## Set Up Environment

1. create an Anaconda environment, with a name e.g. memsum
   
   **Note**: Without further notification, the following commands need to be run in the working directory where this jupyter notebook is located.
   ```bash
   conda create -n memsum python=3.10
   ```
2. activate this environment
   ```bash
   source activate memsum
   ```
   
3. Install pytorch (GPU version). 
   ```bash
   pip install torch torchvision torchaudio
   ```
4. Install dependencies via pip
   ```bash
   pip install -r requirements.txt
   ```
   
Note: If you are runing this notebook on google Colab with GPU runtime, step 1, 2 and 3 are not needed.


# Preprocessing Custom data

Suppose that you have already splitted the training / validation and  test set:

The training data is now stored in a .jsonl file that contains a list of json info, one line for one training instance. Each json (or dictonary) contains two keys: 

1. "text": the value for which is a python list of sentences, this represents the document you want to summarize;
2. "summary": the value is also a list of sentences. If represent the ground-truth summary. Because the summary can contain multiple sentences, so we store them as a list.

The same for the validation file and the testing file. 



```python
import json
train_corpus = [ json.loads(line) for line in open("data/custom_data/train_raw_without_high_rouge_indices_and_scores.jsonl") ]

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



```python

```

If you have your own data, process them into the same structure then put them into the data/ folder

The next thing we need to do is to create high-ROUGE episodes for the training set, as introduced in the paper: https://aclanthology.org/2022.acl-long.450/:

Note: 
* The '!' in the following commands is only needed when running shell command in a jupyter notebook
* Thie script run multi-process tasks. You can change -n_processes to a large value on a machine with powerful multi-core CPUs.


```python
!cd data_preprocessing; python get_high_rouge_episodes_mp.py -input_corpus_file_name ../data/custom_data/train_raw_without_high_rouge_indices_and_scores.jsonl -output_corpus_file_name ../data/custom_data/train.jsonl -beamsearch_size 2 -n_processes 10

```

    70it [00:06, 11.10it/s]]
    finished!
    30it [00:06,  4.53it/s]
    finished!
    50it [00:06,  7.45it/s]
    finished!
    90it [00:06, 13.23it/s] 
    finished!
    10it [00:07,  1.38it/s]
    finished!
    20it [00:07,  2.53it/s]
    finished!
    40it [00:07,  5.01it/s]
    finished!
    80it [00:08,  9.94it/s]
    finished!
    60it [00:08,  7.35it/s]
    finished!
    100it [00:08, 11.28it/s]
    finished!


# Download pretrained word embedding


```python
from huggingface_hub import snapshot_download
## download the pretrained glove word embedding (200 dimension)
snapshot_download('nianlong/memsum-word-embedding', local_dir = "model/word_embedding" )
```

# Start Training

Note:
1. you need to switch to the folder src/MemSum_Full;
2. You can specify the path to training and validation set, the model_folder (where you want to store model checkpoints) and the log_folder (where you want to store the log info), and other parameters. 
3. You can provide the absolute path, or relative path, as shown in the example code below.
4. n_device means the number of available GPUs
5. set save_every and validate_every to 0 will make the training script save and evaluate the model only at the end of each epoch.


```python
!cd src; python train.py -training_corpus_file_name ../data/custom_data/train.jsonl -validation_corpus_file_name ../data/custom_data/val.jsonl -model_folder ../model/memsum-custom-data -log_folder ../log/memsum-custom-data -vocabulary_file_name ../model/word_embedding/vocabulary_200dim.pkl -pretrained_unigram_embeddings_file_name ../model/word_embedding/unigram_embeddings_200dim.pkl -max_seq_len 100 -max_doc_len 500 -num_of_epochs 10 -save_every 0 -validate_every 0 -n_device 1 -batch_size_per_device 4 -max_extracted_sentences_per_document 7 -moving_average_decay 0.999 -p_stop_thres 0.6

```

    100it [00:00, 15554.62it/s]
    100it [00:00, 19677.71it/s]
    24it [00:11,  2.16it/s]Starting validation ...
    [current_batch: 00025] val: 0.0000, 0.0000, 0.0000
    25it [00:15,  1.58it/s]
    24it [00:11,  2.15it/s]Starting validation ...
    [current_batch: 00050] val: 0.3731, 0.1170, 0.3347
    25it [00:18,  1.36it/s]
    24it [00:11,  2.15it/s]Starting validation ...
    [current_batch: 00075] val: 0.3766, 0.1242, 0.3382
    25it [00:18,  1.37it/s]
    24it [00:11,  2.14it/s][current_batch: 00100] loss: 0.508, learning rate: 0.000100
    Starting validation ...
    [current_batch: 00100] val: 0.3781, 0.1259, 0.3404
    25it [00:18,  1.35it/s]
    24it [00:11,  2.12it/s]Starting validation ...
    [current_batch: 00125] val: 0.3731, 0.1225, 0.3356
    25it [00:18,  1.36it/s]
    24it [00:11,  2.13it/s]Starting validation ...
    [current_batch: 00150] val: 0.3717, 0.1225, 0.3351
    25it [00:18,  1.36it/s]
    24it [00:11,  2.14it/s]Starting validation ...
    [current_batch: 00175] val: 0.3789, 0.1303, 0.3415
    25it [00:18,  1.35it/s]
    24it [00:11,  2.13it/s][current_batch: 00200] loss: 0.503, learning rate: 0.000100
    Starting validation ...
    [current_batch: 00200] val: 0.3803, 0.1305, 0.3420
    25it [00:18,  1.35it/s]
    24it [00:11,  2.12it/s]Starting validation ...
    [current_batch: 00225] val: 0.3792, 0.1286, 0.3395
    25it [00:18,  1.35it/s]
    24it [00:11,  2.14it/s]Starting validation ...
    [current_batch: 00250] val: 0.3796, 0.1304, 0.3409
    25it [00:18,  1.35it/s]


# Determine the best checkpoint based on validation loss


```python
! cd src; python get_optimal_batch.py -log_file_name ../log/memsum-custom-data/val.log
```

    batch: 200 scores: ('0.3803', '0.1305', '0.3420')


So the best checkpoint is model_batch_200.pt

# Testing trained model on custom dataset


```python
from src.summarizer import MemSum
from tqdm import tqdm
from rouge_score import rouge_scorer
import json
import numpy as np
```


```python
rouge_cal = rouge_scorer.RougeScorer(['rouge1','rouge2', 'rougeLsum'], use_stemmer=True)

memsum_custom_data = MemSum(  "model/memsum-custom-data/model_batch_200.pt", 
                  "model/word_embedding/vocabulary_200dim.pkl", 
                  gpu = 0 ,  max_doc_len = 500  )
```


```python
test_corpus_custom_data = [ json.loads(line) for line in open("data/custom_data/test.jsonl")]
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

    100%|███████████████████████████████████████████████████████████████| 100/100 [00:08<00:00, 11.23it/s]





    array([0.40608886, 0.14885996, 0.36723294])




```python

```


```python
document = test_corpus_custom_data[10]["text"]
extracted_summary = memsum_custom_data.extract( [ document ], 
                                   p_stop_thres = 0.6, 
                                   max_extracted_sentences_per_document = 7
                                  )[0]
extracted_summary
```




    ['male macroprolactinomas ( mprl ) are usually revealed by headaches , visual troubles and gonadal insufficiency .',
     'suppurative meningitis ( sm ) , a life - threatening condition , is scarcely observed in subjects with macro tumors secreting prolactin ( prl ) and in other pituitary tumors ( pt ) .',
     'however , in some very rare cases it can be a primary presentation or appear after radiotherapy or medical treatment used for tumors destroying the sellar floor and/or the skull base .',
     'our aim was to analyze sm frequency among male mprl deemed to be very invasive tumors , to report our cases and analyze the circumstances under which the dangerous neurological complication appeared .',
     'this destruction leads to cerebral spinal fluid ( csf ) leak , which can act as an entry portal for organisms predisposing to meningitis .',
     'in this retrospective study , we analyzed 82 subjects with mprl to look for symptoms , clinical signs and biological proof of sm .',
     'the described cases emphasize the necessity of an early diagnosis and treatment of large and invasive pt , especially male mprl . medical treatment , which is now the gold standard for prolactinomas ,']




```python

```


```python

```
