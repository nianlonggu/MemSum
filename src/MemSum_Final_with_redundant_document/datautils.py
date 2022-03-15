import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

from tqdm import tqdm


## the corpus has been preprocessed, so here only lower is needed
## all digits are kept, since sent2vec unigram embedding has digit embedding
## no stemming, no lemmatization
class SentenceTokenizer:
    def __init__(self ):
        pass
    def tokenize(self, sen ):
        return sen.lower()

class Vocab:
    def __init__(self, words, eos_token = "<eos>", pad_token = "<pad>", unk_token = "<unk>" ):
        self.words = words
        self.index_to_word = {}
        self.word_to_index = {}
        for idx in range( len(words) ):
            self.index_to_word[ idx ] = words[idx]
            self.word_to_index[ words[idx] ] = idx
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.eos_index = self.word_to_index[self.eos_token]
        self.pad_index = self.word_to_index[self.pad_token]

        self.tokenizer = SentenceTokenizer()   

    def index2word( self, idx ):
        return self.index_to_word.get( idx, self.unk_token)
    def word2index( self, word ):
        return self.word_to_index.get( word, -1 )
    # The sentence needs to be tokenized 
    def sent2seq( self, sent, max_len = None , tokenize = True):
        if tokenize:
            sent = self.tokenizer.tokenize(sent)
        seq = []
        for w in sent.split():
            if w in self.word_to_index:
                seq.append( self.word2index(w) )
        if max_len is not None:
            if len(seq) >= max_len:
                seq = seq[:max_len -1]
                seq.append( self.eos_index )
            else:
                seq.append( self.eos_index )
                seq += [ self.pad_index ] * ( max_len - len(seq) )
        return seq
    def seq2sent( self, seq ):
        sent = []
        for i in seq:
            if i == self.eos_index or i == self.pad_index:
                break
            sent.append( self.index2word(i) )
        return " ".join(sent)

class ExtractionTrainingDataset(Dataset):
    def __init__( self,  corpus, vocab , max_seq_len , max_doc_len  ):
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.max_doc_len = max_doc_len
        ## corpus is a list 
        self.corpus = corpus

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__( self, idx ):

        doc_data = self.corpus[idx]
        sentences = doc_data["text"]
        indices = doc_data["indices"]
        scores = np.array( doc_data["score"] )
        summary = doc_data["summary"]

        ## add redundancy
        redundant_text = []
        for sen in sentences:
            redundant_text+=[ sen, sen ]
            
        indices_for_redundant_text = []
        for idx_list in indices:
            indices_for_redundant_text.append( list( map( lambda x: 2*x + np.random.choice(2) , idx_list ) )  )
        
        sentences = redundant_text
        indices = indices_for_redundant_text


        num_sentences_in_doc = len( sentences )

        ### This is for RL training
        rand_idx = np.random.choice( len(indices) )
        valid_sen_idxs = np.array( indices[ rand_idx ] )

        np.random.shuffle( valid_sen_idxs )

        valid_sen_idxs = valid_sen_idxs[ valid_sen_idxs < num_sentences_in_doc ]
        selected_y_label = np.zeros( num_sentences_in_doc )
        selected_y_label[ valid_sen_idxs ] = 1
        selected_score = scores[ rand_idx ]
        

        valid_sen_idxs = valid_sen_idxs[:self.max_doc_len]
        valid_sen_idxs = np.array(valid_sen_idxs.tolist() + [-1] * ( self.max_doc_len - len(valid_sen_idxs)))

        if num_sentences_in_doc > self.max_doc_len:
            selected_y_label = selected_y_label[:self.max_doc_len]
            sentences = sentences[:self.max_doc_len]
        else:
            selected_y_label = np.array( selected_y_label.tolist() + [0] * (self.max_doc_len - num_sentences_in_doc) )
            sentences += [""] * ( self.max_doc_len - num_sentences_in_doc )
        
        doc_mask = np.array(  [ True if sen.strip() == "" else False for sen in  sentences   ]  )

        seqs = [  self.vocab.sent2seq( sen, self.max_seq_len ) for sen in sentences ]
        seqs = np.asarray( seqs )

        summary = summary[:self.max_doc_len]
        if len(summary) < self.max_doc_len:
            summary = summary + [""] * ( self.max_doc_len - len(summary) )
        summary_seq = []
        for summary_sen in summary:
            summary_seq.append( np.array( self.vocab.sent2seq( summary_sen, self.max_seq_len ) ) )
        summary_seq = np.asarray(summary_seq)

        return seqs, doc_mask, selected_y_label, selected_score, summary_seq, valid_sen_idxs


class ExtractionValidationDataset(Dataset):
    def __init__( self,  corpus, vocab , max_seq_len , max_doc_len  ):
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.max_doc_len = max_doc_len
        ## corpus is a list 
        self.corpus = corpus

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__( self, idx ):

        doc_data = self.corpus[idx]
        sentences = doc_data["text"]
        summary = doc_data["summary"]


        ## add redundancy
        redundant_text = []
        for sen in sentences:
            redundant_text+=[ sen, sen ]
        sentences = redundant_text


        num_sentences_in_doc = len( sentences )

        if num_sentences_in_doc > self.max_doc_len:
            sentences = sentences[:self.max_doc_len]
        else:
            sentences += [""] * ( self.max_doc_len - num_sentences_in_doc )
            
        doc_mask = np.array(  [ True if sen.strip() == "" else False for sen in  sentences   ]  )
            
        seqs = [  self.vocab.sent2seq( sen, self.max_seq_len ) for sen in sentences ]
        seqs = np.asarray( seqs )

        summary = summary[:self.max_doc_len]
        if len(summary) < self.max_doc_len:
            summary = summary + [""] * ( self.max_doc_len - len(summary) )
        summary_seq = []
        for summary_sen in summary:
            summary_seq.append( np.array( self.vocab.sent2seq( summary_sen, self.max_seq_len ) ) )
        summary_seq = np.asarray(summary_seq)

        return seqs, doc_mask, summary_seq
