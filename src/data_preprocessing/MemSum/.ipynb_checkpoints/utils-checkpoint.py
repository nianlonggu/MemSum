import numpy as np

import nltk
from nltk.tokenize import  RegexpTokenizer
from nltk.stem import SnowballStemmer
from functools import lru_cache



class SentenceTokenizer:
    def __init__(self ):
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stemmer = SnowballStemmer("english")

    @lru_cache(100000)
    def stem( self, w ):
        return self.stemmer.stem(w)
    
    def tokenize(self, sen ):
        sen =  [ self.stem(w) for w in self.tokenizer.tokenize( sen.lower() )   ]
        return sen

class Vocab:
    def __init__(self ):
        self.word_to_index = {}
        self.vocab_size = 0
        self.tokenizer = SentenceTokenizer()
    
    def sent2seq( self, sent ):
        seq = []
        words = self.tokenizer.tokenize( sent )
        for w in words:
            if w not in self.word_to_index:
                self.word_to_index[w] = self.vocab_size
                self.vocab_size +=1
            seq.append( self.word_to_index[w] )
        return seq


def fast_rouge_score( ref, hyp, n_gram_list = [1,2], history_results = None ):
    # ref and hyp are lists of word indices
    ref = np.array(ref)
    hyp = np.array(hyp)

    
    results = {}
    for n in n_gram_list:
        if history_results is None:

            ref_ngram = np.concatenate( [ ref[offset: len(ref) -( n-offset )+1  ][:,np.newaxis] for offset in range(n)  ], axis = 1 )
            hyp_ngram = np.concatenate( [ hyp[offset: len(hyp) -( n-offset )+1  ][:,np.newaxis] for offset in range(n)  ], axis = 1 )
        
            if len(ref_ngram) == 0:
                results[ "rouge%d"%(n)] = { "precision":0.0,"recall":0.0,"fmeasure":0.0 }
                continue

            unique_ref_ngram = np.unique( ref_ngram, axis = 0 )
            unique_ref_ngram_expanded_for_ref = unique_ref_ngram[:,np.newaxis,:].repeat( ref_ngram.shape[0], axis = 1 )
            n_match_in_ref = np.all(unique_ref_ngram_expanded_for_ref == ref_ngram[np.newaxis,:,:], axis = 2).sum(1)

            unique_ref_ngram_expanded_for_hyp = unique_ref_ngram[:,np.newaxis,:].repeat( hyp_ngram.shape[0], axis = 1 )
            n_match_in_hyp = np.all(unique_ref_ngram_expanded_for_hyp == hyp_ngram[np.newaxis,:,:], axis = 2).sum(1)
            
            n_hyp = hyp_ngram.shape[0]
        else:
            history_results = history_results.copy()
            ## in this case, we assume ref is the same as before, so we directly load the necessary array from the history
            if "ref_ngram" not in history_results["rouge%d"%(n)] or \
                len(history_results["rouge%d"%(n)]["ref_ngram"]) == 0:
                results[ "rouge%d"%(n)] = { "precision":0.0,"recall":0.0,"fmeasure":0.0 }
                continue    

            ref_ngram = history_results["rouge%d"%(n)]["ref_ngram"]
            hyp_ngram = np.concatenate( [ hyp[offset: len(hyp) -( n-offset )+1  ][:,np.newaxis] for offset in range(n)  ], axis = 1 )
            
            # if len(ref_ngram) == 0:
            #     results[ "rouge%d"%(n)] = { "precision":0.0,"recall":0.0,"fmeasure":0.0 }
            #     continue

            unique_ref_ngram = history_results["rouge%d"%(n)]["unique_ref_ngram"]
            n_match_in_ref = history_results["rouge%d"%(n)]["n_match_in_ref"]

            unique_ref_ngram_expanded_for_hyp = unique_ref_ngram[:,np.newaxis,:].repeat( hyp_ngram.shape[0], axis = 1 )
            n_match_in_hyp = np.all(unique_ref_ngram_expanded_for_hyp == hyp_ngram[np.newaxis,:,:], axis = 2).sum(1)
            n_match_in_hyp = n_match_in_hyp + history_results["rouge%d"%(n)]["n_match_in_hyp"]
            n_hyp = hyp_ngram.shape[0] + history_results["rouge%d"%(n)]["n_hyp"]


        n_common = np.minimum(n_match_in_ref, n_match_in_hyp )
        p = n_common.sum() / (n_hyp+1e-12)
        r = n_common.sum() / ref_ngram.shape[0]
        f = 2/( 1/(r+1e-12) + 1/(p+1e-12) )
        
        results[ "rouge%d"%(n)] = { "precision":p,"recall":r,"fmeasure":f, 
                                    "ref_ngram":ref_ngram,
                                    "unique_ref_ngram":unique_ref_ngram,
                                    "n_match_in_ref":n_match_in_ref,
                                    "n_match_in_hyp":n_match_in_hyp,
                                    "n_hyp":n_hyp
                                     }
        
    return  results


def get_score( hyp, ref, n_gram_list =[1,2], history_results = None, metric = "fmeasure" ):
    res = fast_rouge_score( ref, hyp, n_gram_list, history_results )
    score = np.mean([ res["rouge%d"%(n)][metric] for n in  n_gram_list ])
    return score, res

def get_items( a_list, indices ):
    return [ a_list[i] for i in indices ]

def join_items( items ):
    res = []
    for item in items:
        res += item
    return res

## by default, document is a list of seqs, summary is single seq
def greedy_extract_core(document, summary, extracted_indices, beamsearch_size, max_num_extracted_sentences, max_num_extractions , candidate_extractions, epsilon, n_gram_list =[1,2], metric = "fmeasure"):
    if max_num_extractions is not None and len(candidate_extractions)>=max_num_extractions:
        return
    current_summary = join_items(get_items(document, extracted_indices ))
    current_score, history_results =  get_score(current_summary , summary, n_gram_list = n_gram_list, metric = metric )
    
    if len(extracted_indices) >= max_num_extracted_sentences:
        candidate_extractions.append(  [ extracted_indices, current_score  ] )
        return
    
    remaining_indices = list(set( np.arange(len(document)) ) - set(extracted_indices))
    if len(remaining_indices) == 0:
        if len(extracted_indices) == 0:
            return
        candidate_extractions.append(  [ extracted_indices, current_score  ] )
        return
    
    
    new_scores =[]
    summary_set = set(summary)
    for i in  remaining_indices:
        new_scores.append( get_score( document[i], summary, history_results = history_results, n_gram_list = n_gram_list, metric = metric )[0] )
    new_scores = np.array( new_scores )

    gain_scores = new_scores-current_score
    
    if not np.any( gain_scores>epsilon ):
        if len(extracted_indices) == 0:
            return
        candidate_extractions.append(  [ extracted_indices, current_score  ] )
        return
    
    new_poses = np.argsort( -gain_scores )[:beamsearch_size]
    for pos in new_poses:
        if gain_scores[pos]>0:
            idx = int(remaining_indices[pos])
            greedy_extract_core(document, summary, extracted_indices.copy() + [idx], beamsearch_size,max_num_extracted_sentences, max_num_extractions , candidate_extractions, epsilon, n_gram_list = n_gram_list, metric = metric)    

            
def greedy_extract( document, summary, beamsearch_size=1 , max_num_extracted_sentences = None, max_num_extractions = 20 , epsilon = 0.001, n_gram_list =[1,2], metric = "fmeasure" ):
    vocab  = Vocab()
    document_seq = [ vocab.sent2seq(sen) for sen in document  ]
    summary_seq = [ vocab.sent2seq(sen) for sen in summary  ]
    extracted_indices = []
    candidate_extractions = []
    
    if max_num_extracted_sentences is None or max_num_extracted_sentences <= 0 :
        max_num_extracted_sentences =  len( document )
    
    greedy_extract_core(document_seq, join_items( summary_seq ) , extracted_indices, beamsearch_size, max_num_extracted_sentences, max_num_extractions ,candidate_extractions, epsilon, n_gram_list = n_gram_list, metric = metric)     

    candidate_extractions.sort( key = lambda x :-x[1] )
    existing_extraction_indices = set()
    cleaned_candidate_extractions = []
    for extraction in candidate_extractions:
        extraction_indice = "-".join( map( str, sorted(extraction[0]) ) )
        if not extraction_indice in existing_extraction_indices:
            existing_extraction_indices.add(extraction_indice)
            cleaned_candidate_extractions.append(extraction)  
    
    if len(cleaned_candidate_extractions) == 0:
        cleaned_candidate_extractions = [ [[], 0.0] ]
    
    return cleaned_candidate_extractions
    
    
    