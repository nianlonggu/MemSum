# MemSum
Code for ACL 2022 paper: MemSum: Extractive Summarization of Long Documents Using Multi-Step Episodic Markov Decision Processes.

## System Info
Tested on Ubuntu 20.04 and Ubuntu 18.04

## Step 1: Set up environment
1. create an Anaconda environment, with a name e.g. memsum
   
   **Note**: Without further notification, the following commands need to be run in the working directory where this jupyter notebook is located.
   ```bash
   conda create -n memsum python=3.7
   ```
2. activate this environment
   ```bash
   source activate memsum
   ```
## Step 2: Install dependencies, download word embeddings and load them to pretrained model
1. Install dependencies via pip
   ```bash
   pip install -r requirements.txt
   ```
2. Install pytorch (GPU version). (Here the correct cuda version need to be specified, here we used torch version==1.11.0)
   ```bash
   conda install pytorch cudatoolkit=11.3 -c pytorch -y
   ```
3. Download pretrained word embedding.

   We provide a trained MemSum model on PubMed dataset. In order to use this model, we need to download the pretrained GLOVE word embedding from the official website and add them to MemSum using the following script. This command takes time, as we need to first download and the unzip GloVe embeddings.
   ```bash
   python download_and_load_word_embedding.py
   ```
## Step 3: Testing trained model on a given dataset
For example, the following command test the performance of the full MemSum model, on the Pubmed's test set. The model is evaluated by ROUGE 1/2/L's precision, recall and F1 scores.
```bash
python my_test.py -model_type MemSum_Final -summarizer_model_path model/MemSum_Final/pubmed_full/200dim/final/model.pt -vocabulary_path model/glove/vocabulary_200dim.pkl -corpus_path data/pubmed_full/test_PUBMED.jsonl -gpu 0 -max_extracted_sentences_per_document 7 -p_stop_thres 0.6 -output_file results/MemSum_Final/pubmed_full/200dim/test_results.txt  -max_doc_len 500 -max_seq_len 100
```
Here we provided the trained MemSum on the PubMed dataset and the first 100 training/validation/testing examples for the PubMed, arXiv and GovReport datasets. Other trained models and full datasets used in our experiments will be released soon.

## Step 4: Use the pretrained summarizer as a module in python scripts
1. load the full MemSum model
   ```python
   from my_summarizers import ExtractiveSummarizer_MemSum_Final
   memsum_model = ExtractiveSummarizer_MemSum_Final( 
                "model/MemSum_Final/pubmed_full/200dim/final/model.pt",
                "model/glove/vocabulary_200dim.pkl",  
                gpu = 0,
                embed_dim = 200,
                max_doc_len  = 500,
                max_seq_len = 100
                )
   ```
2. Get a document to be summarized

   The format of the document to be summarized is a list of sentences
   ```python
   import json
   database = [ json.loads(line) for line in open( "data/pubmed_full/test_PUBMED.jsonl" ).readlines() ]
   pos = 6
   document = database[pos]["text"]
   gold_summary =  database[pos]["summary"]
   print(document[:5])
   ```
   ```
   ['the family is the cornerstone of human social support network and its presence is essential in everyone s life . changes inevitably occur in families with illness and hospitalization of a family member . in other words , among the sources of stress for families are accidents leading to hospitalization particularly intensive care unit ( icu ) .', 'statistics show that 8% of hospital beds in the united states are occupied by the intensive care units .', 'stress in the family while the patient is in the icu can disrupt the harmony power of the family members and finally , it may causes disturbances in the support of the patient .', 'in addition to the various sources of stress in intensive care units such as the patient s fear of death , financial problems , lack of awareness about the environment and etc . , their satisfaction level is another important source of stress for the patient s family .', 'today , the family needs of hospitalized patients in the icu are summarized in five sections .']
   ```
   The gold summary is the abstract of the corresponding paper to which the document belongs. Here we get an example from the test set of the PubMed dataset
   ```python
   print(gold_summary)
   ```
   ```
   ['background : since the family is a social system , the impairment in each of its component members may disrupt the entire family system .', 'one of the stress sources for families is accidents leading to hospitalization particularly in the intensive care unit ( icu ) . in many cases ,', 'the families needs in patient care are not met that cause dissatisfaction . since the nurses spend a lot of time with patients and their families , they are in a good position to assess their needs and perform appropriate interventions .', 'therefore , this study was conducted to determine the effectiveness of nursing interventions based on family needs on family satisfaction level of hospitalized patients in the neurosurgery icu.materials and methods : this clinical trial was conducted in the neurosurgery icu of al - zahra hospital , isfahan , iran in 2010 .', 'sixty four families were selected by simple sampling method and were randomly placed in two groups ( test and control ) using envelopes . in the test group ,', 'some interventions were performed to meet their needs . in the control group ,', 'the routine actions were only carried out .', 'the satisfaction questionnaire was completed by both groups two days after admission and again on the fourth day.findings:both of the intervention and control groups were compared in terms of the mean satisfaction scores before and after intervention .', 'there was no significant difference in mean satisfaction scores between test and control groups before the intervention .', 'the mean satisfaction score significantly increased after the intervention compared to the control group.conclusions:nursing interventions based on family needs of hospitalized patients in the icu increase their satisfaction .', 'attention to family nursing should be planned especially in the icus .']
   ```
3. Extractively summarize the document using MemSum
   ```python
   extracted_summary = memsum_model.extract( [ document ], p_stop_thres=0.6, max_extracted_sentences_per_document= 7, return_sentence_position= False )[0]
   print(extracted_summary)
   ```
   ```
   ['the purpose of the study was to determine the effectiveness of nursing interventions based on family needs on family satisfaction level of hospitalized patients in the neurosurgery intensive care unit of al - zahra hospital in 2010 .', 'in this study , it was shown that the use of nursing interventions based on family needs ( confidence , support , information , proximity and convenience ) had significant impact on the family satisfaction of the patient hospitalized in intensive care unit .', 'the statistical research community was the families of hospitalized patients in neurosurgery intensive care unit of al - zahra ( sa ) hospital , isfahan , iran from may to september 2010 .', 'the aim of this study was to analyze the satisfaction of the families of icu patients .', 'the results of the present study showed that the nursing interventions based on the family needs increased the patient s family satisfaction in the neurosurgery intensive care unit of al - zahra hospital .', 'comparison of mean satisfaction score ( 100 * ) of participants in the intervention and control groups the mean of satisfaction score changes of the studied subjects in the intervention and control groups after intervention', 'the mean satisfaction score in the intervention group after the intervention was significantly higher than before the intervention ( p < 0.001 ) .']
   ```
4. Evaluate the extracted summary via ROUGE scores
   ```python
   from rouge_score import rouge_scorer
   rouge_cal = rouge_scorer.RougeScorer(['rouge1','rouge2', 'rougeLsum'], use_stemmer=True)
   print(rouge_cal.score( "\n".join(gold_summary), "\n".join(extracted_summary)  ))
   ```
   ```
   {'rouge1': Score(precision=0.6517412935323383, recall=0.4833948339483395, fmeasure=0.5550847457627119), 'rouge2': Score(precision=0.36, recall=0.26666666666666666, fmeasure=0.30638297872340425), 'rougeLsum': Score(precision=0.6069651741293532, recall=0.45018450184501846, fmeasure=0.5169491525423728)}
   ```
## Step 5: Training model from script
For example, if we want to train the full MemSum model on the PubMed dataset, we first change working directory to "src/MemSum_Final/", then run the python script "train.py". The train.py takes one parameter: config_file_path, which is the path to the training configuration file.

In the configuration file there are detailed list of key-value pairs that configure the training procedure. For example, the number of GPU devices, batch size, learning rate, etc. 

**Note** Here we used 4 GPUs, so in the config_file_path the value for n_device is 4. When using different number of GPUs, such as 1, the value of n_device needs to be changed accordingly.
   ```bash
   cd src/MemSum_Final/; python train.py -config_file_path config/pubmed_full/200dim/run0/training.config
   ```
## Additional Info
We provide the human evaluation raw data obtained from two human evaluation experiments as discussed in the main paper. Each line in the .jsonl file contains a record of a single evaluation, including: 1) document to be summarized, 2) gold summary, 3) summaries produced by two models, and 4) human evaluation ranking results of both summaries. The data is available in data/ folder.
 
## References
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
