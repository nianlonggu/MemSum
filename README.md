# MemSum: Extractive Summarization of Long Documents Using Multi-Step Episodic Markov Decision Processes

<a href="https://colab.research.google.com/github/nianlonggu/MemSum/blob/main/Data_processing_training_and_testing_for_MemSum.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Code for ACL 2022 paper on the topic of long document extractive summarization: [MemSum: Extractive Summarization of Long Documents Using Multi-Step Episodic Markov Decision Processes](https://aclanthology.org/2022.acl-long.450/).

## Update 04-11-2022

Add additional information on how to call the greedy extract method: both the text and the summary are a list of sentences.

## Update 18-10-2022

Added the instruction on how to process training / validation / test set of custom data and step-by-step intro on how to train and test the model.

We provided a google colab notebook:
https://github.com/nianlonggu/MemSum/blob/main/Data_processing_training_and_testing_for_MemSum.ipynb
WIthin this nodebook you can simply run the codes line by line for the whole data preprocessing training and testing pipeline.

You can also open it by clicking:
<a href="https://colab.research.google.com/github/nianlonggu/MemSum/blob/main/Data_processing_training_and_testing_for_MemSum.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>



## Update 28-07-2022

1. Uploaded processed datasets used in this paper:

The main datasets used in this project include PubMed, PubMed(truncated), arXiv and GovReport.
Detailed statistics about these datasets are available in the paper.

2. Uploaded trained model checkpoints used for the evaluation:

The code for evaluation is shown below

3. Provided the scripts for obtaining the Oracle extraction and for creating High-ROUGE Episodes for training:

The script is available at src/data_preprocessing/



### System Info
Tested on Ubuntu 20.04 and Ubuntu 18.04

### Step 0: Download all datasets and trained model checkpoints

Download the datasets and trained model from the [Google Drive LINK](https://drive.google.com/drive/folders/1X1KNkP-BW_exuTYD94BnlwWs9g0ajJ78?usp=sharing).

Put the "data/" folder and the "model/" folder under the same folder as the "src/" folder. The final structure looks like:

```
MemSum
├── src
│   ├── data_preprocessing
│   │   ├── MemSum
│   │       ├──create_dataset_faster.py
│   │       ├──create_dataset_faster.sh
│   │       ├──merge_files.py
│   ├── MemSum_Full
│   │   ├── datautils.py
│   │   ├── get_optimal_batch.py
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── utils.py
├── data
│   ├── arxiv
│   ├── gov-report
│   ├── pubmed
│   ├── pubmed_truncated
├── model
│   ├── glove
│   ├── MemSum_Full
│   │   ├── arxiv
│   │   ├── gov-report
│   │   ├── pubmed
│   │   ├── pubmed_truncated
├── summarizers.py
└── README.md
```

### Step 1: Set up environment
1. create an Anaconda environment, with a name e.g. memsum
   
   **Note**: Without further notification, the following commands need to be run in the working directory where this jupyter notebook is located.
   ```bash
   conda create -n memsum python=3.7
   ```
2. activate this environment
   ```bash
   source activate memsum
   ```
### Step 2: Install dependencies, download word embeddings and load them to pretrained model
1. Install dependencies via pip
   ```bash
   pip install -r requirements.txt
   ```
2. Install pytorch (GPU version). (Here the correct cuda version need to be specified, here we used torch version==1.11.0)
   ```bash
   conda install pytorch cudatoolkit=11.3 -c pytorch -y
   ```

### Step 3: Testing trained model on a given dataset
For example, the following command test the performance of the full MemSum model. Berfore runing these codes, make sure current working directory is the main directory "MemSum/" where the .py file summarizers.py is located.


```python
from summarizers import MemSum
from tqdm import tqdm
from rouge_score import rouge_scorer
import json
import numpy as np
```


```python
rouge_cal = rouge_scorer.RougeScorer(['rouge1','rouge2', 'rougeLsum'], use_stemmer=True)

memsum_pubmed = MemSum(  "model/MemSum_Full/pubmed/200dim/run0/model_batch_65000.pt", 
                  "model/glove/vocabulary_200dim.pkl", 
                  gpu = 0 ,  max_doc_len = 500  )

memsum_pubmed_truncated = MemSum(  "model/MemSum_Full/pubmed_truncated/200dim/run0/model_batch_49000.pt", 
                  "model/glove/vocabulary_200dim.pkl", 
                  gpu = 0 ,  max_doc_len = 50  )

memsum_arxiv = MemSum(  "model/MemSum_Full/arxiv/200dim/run0/model_batch_37000.pt", 
                  "model/glove/vocabulary_200dim.pkl", 
                  gpu = 0 ,  max_doc_len = 500  )

memsum_gov_report = MemSum(  "model/MemSum_Full/gov-report/200dim/run0/model_batch_22000.pt", 
                  "model/glove/vocabulary_200dim.pkl", 
                  gpu = 0 ,  max_doc_len = 500  )
```


```python
test_corpus_pubmed = [ json.loads(line) for line in open("data/pubmed/test_PUBMED.jsonl") ]
test_corpus_pubmed_truncated = [ json.loads(line) for line in open("data/pubmed_truncated/test_PUBMED.jsonl") ]
test_corpus_arxiv = [ json.loads(line) for line in open("data/arxiv/test_ARXIV.jsonl") ]
test_corpus_gov_report = [ json.loads(line) for line in open("data/gov-report/test_GOVREPORT.jsonl") ]
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
evaluate( memsum_pubmed, test_corpus_pubmed, 0.6, 7, rouge_cal )
```

    100%|██████████| 6658/6658 [10:05<00:00, 10.99it/s]

    array([0.49260137, 0.22916328, 0.44415123])

```python
evaluate( memsum_pubmed_truncated, test_corpus_pubmed_truncated, 0.8, 7, rouge_cal )
```

    100%|██████████| 5025/5025 [04:11<00:00, 19.97it/s]

    array([0.43079567, 0.16707743, 0.38297921])

```python
evaluate( memsum_arxiv, test_corpus_arxiv, 0.5, 5, rouge_cal )
```

    100%|██████████| 6440/6440 [08:28<00:00, 12.66it/s]

    array([0.47946925, 0.19970128, 0.42075852])


```python
evaluate( memsum_gov_report, test_corpus_gov_report, 0.6, 22, rouge_cal )
```

    100%|██████████| 973/973 [04:44<00:00,  3.41it/s]

    array([0.59445629, 0.28507926, 0.56677073])

```python

```


### Step 4: Training model from script
For example, if we want to train the full MemSum model on the PubMed dataset, we first change working directory to "src/MemSum_Full/", then run the python script "train.py".

**Note** Here we used 4 GPUs, so n_device is 4.
   ```bash
   python train.py -training_corpus_file_name ../../data/pubmed/train_PUBMED.jsonl -validation_corpus_file_name ../../data/pubmed/val_PUBMED.jsonl -model_folder ../../model/MemSum_Full/pubmed/200dim/run0/ -log_folder ../../log/MemSum_Full/pubmed/200dim/run0/ -vocabulary_file_name ../../model/glove/vocabulary_200dim.pkl -pretrained_unigram_embeddings_file_name ../../model/glove/unigram_embeddings_200dim.pkl -max_seq_len 100 -max_doc_len 500 -num_of_epochs 100 -save_every 1000 -n_device 2 -batch_size_per_device 16 -max_extracted_sentences_per_document 7 -moving_average_decay 0.999 -p_stop_thres 0.6
   ```
<!-- ## Additional Info
We provide the human evaluation raw data obtained from two human evaluation experiments as discussed in the main paper. Each line in the .jsonl file contains a record of a single evaluation, including: 1) document to be summarized, 2) gold summary, 3) summaries produced by two models, and 4) human evaluation ranking results of both summaries. The data is available in data/ folder. -->


### Addition Info: Code for obtaining the greedy summary of a document, and creating High-ROUGE episodes for training the model.

 
```python
from src.data_preprocessing.MemSum.utils import greedy_extract
import json
```

```python
with open("data/pubmed/val_PUBMED.jsonl","r") as f:
    for line in f:
        break
example_data = json.loads(line)
print(example_data.keys())
```
    dict_keys(['summary', 'text', 'sorted_indices'])


We can extract the oracle summary by calling the function greedy_extract and set beamsearch_size = 1
```python
greedy_extract( example_data["text"], example_data["summary"], beamsearch_size = 1 )[0]
```
    [[72, 11, 20, 134, 102, 79, 9, 99, 39, 34, 44], 0.4777551272557634]

Here the first element is a list of sentence indices in the document, the second element is the avarge of Rouge F1 scores.

By setting beamsearch_size = 2 or more, we can extract the high-rouge episodes, a candidate list of extracted sentences' indices and the corresponding scores that can be used for training models

**Note: both example_data["text"] and example_data["summary"] are list of sentences, please do not join the sentence list into a string**

```python
greedy_extract( example_data["text"], example_data["summary"], beamsearch_size = 2 )
```
    [[[72, 11, 20, 134, 102, 79, 9, 99, 39, 34, 44], 0.4777551272557634],
     [[72, 11, 20, 134, 102, 79, 9, 99, 39, 34, 74], 0.4777551272557634],
     [[72, 11, 20, 134, 102, 79, 9, 99, 69, 34, 44], 0.4777551272557634],
     [[72, 11, 20, 134, 102, 79, 9, 99, 69, 34, 74], 0.4777551272557634],
     [[72, 11, 20, 134, 102, 79, 9, 99, 39, 44, 116, 34], 0.4775433538646387],
     [[72, 11, 20, 134, 102, 79, 9, 99, 69, 44, 116, 34], 0.4775433538646387],
     [[72, 11, 20, 134, 102, 79, 9, 69, 95, 99, 83], 0.47290795715372624],
     [[72, 11, 20, 134, 102, 79, 9, 69, 95, 99, 44], 0.47290795715372624],
     [[72, 11, 20, 134, 102, 79, 9, 69, 95, 44, 116, 99], 0.47283962445015093],
     [[72, 11, 20, 134, 102, 79, 9, 69, 95, 44, 116, 34, 99], 0.4726851816645392]]

In the folder src/data_preprocessing/MemSum/ there are scripts that can be directly called to obtain high-rouge episodes which works in parallel.


### References
When using our code or models for your application, please cite the following paper:
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
    pages = "6507--6522",
    abstract = "We introduce MemSum (Multi-step Episodic Markov decision process extractive SUMmarizer), a reinforcement-learning-based extractive summarizer enriched at each step with information on the current extraction history. When MemSum iteratively selects sentences into the summary, it considers a broad information set that would intuitively also be used by humans in this task: 1) the text content of the sentence, 2) the global text context of the rest of the document, and 3) the extraction history consisting of the set of sentences that have already been extracted. With a lightweight architecture, MemSum obtains state-of-the-art test-set performance (ROUGE) in summarizing long documents taken from PubMed, arXiv, and GovReport. Ablation studies demonstrate the importance of local, global, and history information. A human evaluation confirms the high quality and low redundancy of the generated summaries, stemming from MemSum{'}s awareness of extraction history.",
}
```
