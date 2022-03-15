from src.MemSum_Final.model import LocalSentenceEncoder as LocalSentenceEncoder_MemSum_Final
from src.MemSum_Final.model import GlobalContextEncoder as GlobalContextEncoder_MemSum_Final
from src.MemSum_Final.model import ExtractionContextDecoder as ExtractionContextDecoder_MemSum_Final
from src.MemSum_Final.model import Extractor as Extractor_MemSum_Final
from src.MemSum_Final.datautils import Vocab as Vocab_MemSum_Final
from src.MemSum_Final.datautils import SentenceTokenizer as SentenceTokenizer_MemSum_Final

from src.MemSum_wo_history.model import LocalSentenceEncoder as LocalSentenceEncoder_MemSum_wo_history
from src.MemSum_wo_history.model import GlobalContextEncoder as GlobalContextEncoder_MemSum_wo_history
from src.MemSum_wo_history.model import ExtractionContextDecoder as ExtractionContextDecoder_MemSum_wo_history
from src.MemSum_wo_history.model import Extractor as Extractor_MemSum_wo_history
from src.MemSum_wo_history.datautils import Vocab as Vocab_MemSum_wo_history
from src.MemSum_wo_history.datautils import SentenceTokenizer as SentenceTokenizer_MemSum_wo_history



from src.MemSum_with_stop_sentence.model import LocalSentenceEncoder as LocalSentenceEncoder_MemSum_with_stop_sentence
from src.MemSum_with_stop_sentence.model import GlobalContextEncoder as GlobalContextEncoder_MemSum_with_stop_sentence
from src.MemSum_with_stop_sentence.model import ExtractionContextDecoder as ExtractionContextDecoder_MemSum_with_stop_sentence
from src.MemSum_with_stop_sentence.model import Extractor as Extractor_MemSum_with_stop_sentence
from src.MemSum_with_stop_sentence.datautils import Vocab as Vocab_MemSum_with_stop_sentence
from src.MemSum_with_stop_sentence.datautils import SentenceTokenizer as SentenceTokenizer_MemSum_with_stop_sentence



import torch.nn.functional as F
from torch.distributions import Categorical

import pickle
import torch
import numpy as np

from tqdm import tqdm
import json



class ExtractiveSummarizer_MemSum_Final:
    def __init__( self, model_path, vocabulary_path, gpu = None , embed_dim=200, num_heads=8, hidden_dim = 1024, N_enc_l = 2 , N_enc_g = 2, N_dec = 3,  max_seq_len =500, max_doc_len = 100  ):
        with open( vocabulary_path , "rb" ) as f:
            words = pickle.load(f)
        self.vocab = Vocab_MemSum_Final( words )
        vocab_size = len(words)
        self.local_sentence_encoder = LocalSentenceEncoder_MemSum_Final( vocab_size, self.vocab.pad_index, embed_dim,num_heads,hidden_dim,N_enc_l, None )
        self.global_context_encoder = GlobalContextEncoder_MemSum_Final( embed_dim, num_heads, hidden_dim, N_enc_g )
        self.extraction_context_decoder = ExtractionContextDecoder_MemSum_Final( embed_dim, num_heads, hidden_dim, N_dec )
        self.extractor = Extractor_MemSum_Final( embed_dim, num_heads )
        ckpt = torch.load( model_path, map_location = "cpu" )
        self.local_sentence_encoder.load_state_dict( ckpt["local_sentence_encoder"] )
        self.global_context_encoder.load_state_dict( ckpt["global_context_encoder"] )
        self.extraction_context_decoder.load_state_dict( ckpt["extraction_context_decoder"] )
        self.extractor.load_state_dict(ckpt["extractor"])
        
        self.device =  torch.device( "cuda:%d"%(gpu) if gpu is not None and torch.cuda.is_available() else "cpu"  )        
        self.local_sentence_encoder.to(self.device)
        self.global_context_encoder.to(self.device)
        self.extraction_context_decoder.to(self.device)
        self.extractor.to(self.device)
        
        self.sentence_tokenizer = SentenceTokenizer_MemSum_Final()
        self.max_seq_len = max_seq_len
        self.max_doc_len = max_doc_len
    
    def get_ngram(self,  w_list, n = 4 ):
        ngram_set = set()
        for pos in range(len(w_list) - n + 1 ):
            ngram_set.add( "_".join( w_list[ pos:pos+n] )  )
        return ngram_set

    def extract( self, document_batch, p_stop_thres = 0.7, ngram_blocking = False, ngram = 3, return_sentence_position = False, return_sentence_score_history = False, max_extracted_sentences_per_document = 4 ):
        """document_batch is a batch of documents:
        [  [ sen1, sen2, ... , senL1 ], 
           [ sen1, sen2, ... , senL2], ...
         ]
        """
        ## tokenization:
        document_length_list = []
        sentence_length_list = []
        tokenized_document_batch = []
        for document in document_batch:
            tokenized_document = []
            for sen in document:
                tokenized_sen = self.sentence_tokenizer.tokenize( sen )
                tokenized_document.append( tokenized_sen )
                sentence_length_list.append( len(tokenized_sen.split()) )
            tokenized_document_batch.append( tokenized_document )
            document_length_list.append( len(tokenized_document) )

        max_document_length =  self.max_doc_len 
        max_sentence_length =  self.max_seq_len 
        ## convert to sequence
        seqs = []
        doc_mask = []
        
        for document in tokenized_document_batch:
            if len(document) > max_document_length:
                # doc_mask.append(  [0] * max_document_length )
                document = document[:max_document_length]
            else:
                # doc_mask.append(  [0] * len(document) +[1] * ( max_document_length -  len(document) ) )
                document = document + [""] * ( max_document_length -  len(document) )

            doc_mask.append(  [ 1 if sen.strip() == "" else 0 for sen in  document   ] )

            document_sequences = []
            for sen in document:
                seq = self.vocab.sent2seq( sen, max_sentence_length )
                document_sequences.append(seq)
            seqs.append(document_sequences)
        seqs = np.asarray(seqs)
        doc_mask = np.asarray(doc_mask) == 1
        seqs = torch.from_numpy(seqs).to(self.device)
        doc_mask = torch.from_numpy(doc_mask).to(self.device)

        extracted_sentences = []
        sentence_score_history = []
        p_stop_history = []
        
        with torch.no_grad():
            num_sentences = seqs.size(1)
            sen_embed  = self.local_sentence_encoder( seqs.view(-1, seqs.size(2) )  )
            sen_embed = sen_embed.view( -1, num_sentences, sen_embed.size(1) )
            relevance_embed = self.global_context_encoder( sen_embed, doc_mask  )
    
            num_documents = seqs.size(0)
            doc_mask = doc_mask.detach().cpu().numpy()
            seqs = seqs.detach().cpu().numpy()
    
            extracted_sentences = []
            extracted_sentences_positions = []
        
            for doc_i in range(num_documents):
                current_doc_mask = doc_mask[doc_i:doc_i+1]
                current_remaining_mask_np = np.ones_like(current_doc_mask ).astype(np.bool) | current_doc_mask
                current_extraction_mask_np = np.zeros_like(current_doc_mask).astype(np.bool) | current_doc_mask
        
                current_sen_embed = sen_embed[doc_i:doc_i+1]
                current_relevance_embed = relevance_embed[ doc_i:doc_i+1 ]
                current_redundancy_embed = None
        
                current_hyps = []
                extracted_sen_ngrams = set()

                sentence_score_history_for_doc_i = []

                p_stop_history_for_doc_i = []
                
                for step in range( max_extracted_sentences_per_document+1 ) :
                    current_extraction_mask = torch.from_numpy( current_extraction_mask_np ).to(self.device)
                    current_remaining_mask = torch.from_numpy( current_remaining_mask_np ).to(self.device)
                    if step > 0:
                        current_redundancy_embed = self.extraction_context_decoder( current_sen_embed, current_remaining_mask, current_extraction_mask  )
                    p, p_stop, _ = self.extractor( current_sen_embed, current_relevance_embed, current_redundancy_embed , current_extraction_mask  )
                    p_stop = p_stop.unsqueeze(1)
            
            
                    p = p.masked_fill( current_extraction_mask, 1e-12 ) 

                    sentence_score_history_for_doc_i.append( p.detach().cpu().numpy() )

                    p_stop_history_for_doc_i.append(  p_stop.squeeze(1).item() )

                    normalized_p = p / p.sum(dim=1, keepdims = True)

                    stop = p_stop.squeeze(1).item()> p_stop_thres #and step > 0
                    
                    #sen_i = normalized_p.argmax(dim=1)[0]
                    _, sorted_sen_indices =normalized_p.sort(dim=1, descending= True)
                    sorted_sen_indices = sorted_sen_indices[0]
                    
                    extracted = False
                    for sen_i in sorted_sen_indices:
                        sen_i = sen_i.item()
                        if sen_i< len(document_batch[doc_i]):
                            sen = document_batch[doc_i][sen_i]
                        else:
                            break
                        sen_ngrams = self.get_ngram( sen.lower().split(), ngram )
                        if not ngram_blocking or len( extracted_sen_ngrams &  sen_ngrams ) < 1:
                            extracted_sen_ngrams.update( sen_ngrams )
                            extracted = True
                            break
                                        
                    if stop or step == max_extracted_sentences_per_document or not extracted:
                        extracted_sentences.append( [ document_batch[doc_i][sen_i] for sen_i in  current_hyps if sen_i < len(document_batch[doc_i])    ] )
                        extracted_sentences_positions.append( [ sen_i for sen_i in  current_hyps if sen_i < len(document_batch[doc_i])  ]  )
                        break
                    else:
                        current_hyps.append(sen_i)
                        current_extraction_mask_np[0, sen_i] = True
                        current_remaining_mask_np[0, sen_i] = False

                sentence_score_history.append(sentence_score_history_for_doc_i)
                p_stop_history.append( p_stop_history_for_doc_i )

        # if return_sentence_position:
        #     return extracted_sentences, extracted_sentences_positions 
        # else:
        #     return extracted_sentences

        results = [extracted_sentences]
        if return_sentence_position:
            results.append( extracted_sentences_positions )
        if return_sentence_score_history:
            results+=[sentence_score_history , p_stop_history ]
        if len(results) == 1:
            results = results[0]
        
        return results


    def analyze( self, document_batch, extraction_history , p_stop_thres = 0.7, ngram_blocking = False, ngram = 3, return_sentence_position = False, return_sentence_score_history = False ):
        """document_batch is a batch of documents:
        [  [ sen1, sen2, ... , senL1 ], 
           [ sen1, sen2, ... , senL2], ...
         ]
        """
        ## tokenization:
        document_length_list = []
        sentence_length_list = []
        tokenized_document_batch = []
        for document in document_batch:
            tokenized_document = []
            for sen in document:
                tokenized_sen = self.sentence_tokenizer.tokenize( sen )
                tokenized_document.append( tokenized_sen )
                sentence_length_list.append( len(tokenized_sen.split()) )
            tokenized_document_batch.append( tokenized_document )
            document_length_list.append( len(tokenized_document) )

        max_document_length =  self.max_doc_len 
        max_sentence_length =  self.max_seq_len 
        ## convert to sequence
        seqs = []
        doc_mask = []
        
        for document in tokenized_document_batch:
            if len(document) > max_document_length:
                # doc_mask.append(  [0] * max_document_length )
                document = document[:max_document_length]
            else:
                # doc_mask.append(  [0] * len(document) +[1] * ( max_document_length -  len(document) ) )
                document = document + [""] * ( max_document_length -  len(document) )
            doc_mask.append(  [ 1 if sen.strip() == "" else 0 for sen in  document   ] )

            document_sequences = []
            for sen in document:
                seq = self.vocab.sent2seq( sen, max_sentence_length )
                document_sequences.append(seq)
            seqs.append(document_sequences)
        seqs = np.asarray(seqs)
        doc_mask = np.asarray(doc_mask) == 1
        seqs = torch.from_numpy(seqs).to(self.device)
        doc_mask = torch.from_numpy(doc_mask).to(self.device)

        extracted_sentences = []
        sentence_score_history = []
        
        with torch.no_grad():
            num_sentences = seqs.size(1)
            sen_embed  = self.local_sentence_encoder( seqs.view(-1, seqs.size(2) )  )
            sen_embed = sen_embed.view( -1, num_sentences, sen_embed.size(1) )
            relevance_embed = self.global_context_encoder( sen_embed, doc_mask  )
    
            num_documents = seqs.size(0)
            doc_mask = doc_mask.detach().cpu().numpy()
            seqs = seqs.detach().cpu().numpy()
    
            extracted_sentences = []
            extracted_sentences_positions = []
        
            for doc_i in range(num_documents):
                current_doc_mask = doc_mask[doc_i:doc_i+1]
                current_remaining_mask_np = np.ones_like(current_doc_mask ).astype(np.bool) | current_doc_mask
                current_extraction_mask_np = np.zeros_like(current_doc_mask).astype(np.bool) | current_doc_mask
        
                current_sen_embed = sen_embed[doc_i:doc_i+1]
                current_relevance_embed = relevance_embed[ doc_i:doc_i+1 ]
                current_redundancy_embed = None
        
                current_hyps = []
                extracted_sen_ngrams = set()

                sentence_score_history_for_doc_i = []
                
                for step in range( len(extraction_history) ) :
                    current_extraction_mask = torch.from_numpy( current_extraction_mask_np ).to(self.device)
                    current_remaining_mask = torch.from_numpy( current_remaining_mask_np ).to(self.device)
                    if step > 0:
                        current_redundancy_embed = self.extraction_context_decoder( current_sen_embed, current_remaining_mask, current_extraction_mask  )
                    p, p_stop, _ = self.extractor( current_sen_embed, current_relevance_embed, current_redundancy_embed , current_extraction_mask  )
                    p_stop = p_stop.unsqueeze(1)
            
            
                    p = p.masked_fill( current_extraction_mask, 1e-12 ) 

                    sentence_score_history_for_doc_i.append( p.detach().cpu().numpy() )

                    normalized_p = p / p.sum(dim=1, keepdims = True)

                    stop = p_stop.squeeze(1).item()> p_stop_thres #and step > 0
                    
                    #sen_i = normalized_p.argmax(dim=1)[0]
                    _, sorted_sen_indices =normalized_p.sort(dim=1, descending= True)
                    sorted_sen_indices = [extraction_history[step]]  #sorted_sen_indices[0]
                    

                                        
                    sen_i = extraction_history[step]
                    current_extraction_mask_np[0, sen_i] = True
                    current_remaining_mask_np[0, sen_i] = False
                    print(current_extraction_mask_np)
                    print(current_remaining_mask_np)

                return(current_redundancy_embed.detach().cpu().numpy())

        # if return_sentence_position:
        #     return extracted_sentences, extracted_sentences_positions 
        # else:
        #     return extracted_sentences

        results = [extracted_sentences]
        if return_sentence_position:
            results.append( extracted_sentences_positions )
        if return_sentence_score_history:
            results.append( sentence_score_history )
        if len(results)==1:
            results = results[0]
        
        return results




class ExtractiveSummarizer_MemSum_wo_history:
    def __init__( self, model_path, vocabulary_path, gpu = None , embed_dim=200, num_heads=8, hidden_dim = 1024, N_enc_l = 2, N_enc_g = 2, N_dec = 3, max_seq_len =500, max_doc_len = 100  ):
        with open( vocabulary_path , "rb" ) as f:
            words = pickle.load(f)
        self.vocab = Vocab_MemSum_wo_history( words )
        vocab_size = len(words)
        self.local_sentence_encoder = LocalSentenceEncoder_MemSum_wo_history( vocab_size, self.vocab.pad_index, embed_dim,num_heads,hidden_dim,N_enc_l, None )
        self.global_context_encoder = GlobalContextEncoder_MemSum_wo_history( embed_dim, num_heads, hidden_dim, N_enc_g )
        self.extraction_context_decoder = ExtractionContextDecoder_MemSum_wo_history( embed_dim, num_heads, hidden_dim, N_dec )
        self.extractor = Extractor_MemSum_wo_history( embed_dim, num_heads )
        ckpt = torch.load( model_path, map_location = "cpu" )
        self.local_sentence_encoder.load_state_dict( ckpt["local_sentence_encoder"] )
        self.global_context_encoder.load_state_dict( ckpt["global_context_encoder"] )
        self.extraction_context_decoder.load_state_dict( ckpt["extraction_context_decoder"] )
        self.extractor.load_state_dict(ckpt["extractor"])
        
        self.device =  torch.device( "cuda:%d"%(gpu) if gpu is not None and torch.cuda.is_available() else "cpu"  )        
        self.local_sentence_encoder.to(self.device)
        self.global_context_encoder.to(self.device)
        self.extraction_context_decoder.to(self.device)
        self.extractor.to(self.device)
        
        self.sentence_tokenizer = SentenceTokenizer_MemSum_wo_history()
        self.max_seq_len = max_seq_len
        self.max_doc_len = max_doc_len


    def get_ngram(self,  w_list, n = 4 ):
        ngram_set = set()
        for pos in range(len(w_list) - n + 1 ):
            ngram_set.add( "_".join( w_list[ pos:pos+n] )  )
        return ngram_set
    


    def extract( self, document_batch, p_stop_thres = None, ngram_blocking = False, ngram = 3, return_sentence_position = False, return_sentence_score_history = None, max_extracted_sentences_per_document = 4 ):
        """document_batch is a batch of documents:
        [  [ sen1, sen2, ... , senL1 ], 
           [ sen1, sen2, ... , senL2], ...
         ]
        """
        ## tokenization:
        document_length_list = []
        sentence_length_list = []
        tokenized_document_batch = []
        for document in document_batch:
            tokenized_document = []
            for sen in document:
                tokenized_sen = self.sentence_tokenizer.tokenize( sen )
                tokenized_document.append( tokenized_sen )
                sentence_length_list.append( len(tokenized_sen.split()) )
            tokenized_document_batch.append( tokenized_document )
            document_length_list.append( len(tokenized_document) )

        max_document_length =  self.max_doc_len 
        max_sentence_length =  self.max_seq_len 
        ## convert to sequence
        seqs = []
        doc_mask = []
        
        for document in tokenized_document_batch:
            if len(document) > max_document_length:
                # doc_mask.append(  [0] * max_document_length )
                document = document[:max_document_length]
            else:
                # doc_mask.append(  [0] * len(document) +[1] * ( max_document_length -  len(document) ) )
                document = document + [""] * ( max_document_length -  len(document) )
            doc_mask.append(  [ 1 if sen.strip() == "" else 0 for sen in  document   ] )
            document_sequences = []
            for sen in document:
                seq = self.vocab.sent2seq( sen, max_sentence_length )
                document_sequences.append(seq)
            seqs.append(document_sequences)
        seqs = np.asarray(seqs)
        doc_mask = np.asarray(doc_mask) == 1
        seqs = torch.from_numpy(seqs).to(self.device)
        doc_mask = torch.from_numpy(doc_mask).to(self.device)

        
        with torch.no_grad():
            num_sentences = seqs.size(1)
            local_sen_embed  = self.local_sentence_encoder( seqs.view(-1, seqs.size(2) )  )
            local_sen_embed = local_sen_embed.view( -1, num_sentences, local_sen_embed.size(1) )
            global_context_embed = self.global_context_encoder( local_sen_embed, doc_mask  )
    
            num_documents = seqs.size(0)
            doc_mask = doc_mask.detach().cpu().numpy()
            remaining_mask_np = np.ones_like( doc_mask ).astype( np.bool ) | doc_mask
            extraction_mask_np = np.zeros_like( doc_mask ).astype( np.bool ) | doc_mask

            seqs = seqs.detach().cpu().numpy()

            # extraction_context_embed = self.extraction_context_decoder(  local_sen_embed, 
            #                                 torch.from_numpy( remaining_mask_np ).to(self.device), 
            #                                 torch.from_numpy( extraction_mask_np ).to(self.device) )

            extraction_context_embed = None

            p, _, _ = self.extractor(  local_sen_embed, 
                           global_context_embed, 
                           extraction_context_embed , 
                           torch.from_numpy( extraction_mask_np ).to(self.device)  )

            done_list = []
            extracted_sen_ngrams = [ set() for _ in range(num_documents) ]


            for step in range(max_extracted_sentences_per_document):
                extraction_mask = torch.from_numpy( extraction_mask_np ).to(self.device)
        
                p = p.masked_fill( extraction_mask, 1e-12 )  
                normalized_p = p / (p.sum(dim=1, keepdims = True))
        
                done = torch.all( extraction_mask, dim = 1) 
                if len(done_list) > 0:
                    done = torch.logical_or(done_list[-1], done)
                if torch.all( done ):
                    break
            
                
                done_list.append(done)


                # sen_indices = torch.argmax(normalized_p, dim =1)

                _, sorted_sen_indices =normalized_p.sort(dim=1, descending= True)
        
                for doc_i in range( num_documents ):
                    if not done[doc_i]:
                        # sen_i = sen_indices[ doc_i ].item()
                        # remaining_mask_np[doc_i,sen_i] = False
                        # extraction_mask_np[doc_i,sen_i] = True

                        extracted = False
                        for sen_i in sorted_sen_indices[doc_i]:
                            sen_i = sen_i.item()
                            if sen_i< len(document_batch[doc_i]):
                                sen = document_batch[doc_i][sen_i]
                            else:
                                break
                            sen_ngrams = self.get_ngram( sen.lower().split(), ngram )
                            if not ngram_blocking or len( extracted_sen_ngrams[doc_i] &  sen_ngrams ) < 1:
                                extracted_sen_ngrams[doc_i].update( sen_ngrams )
                                extracted = True
                                break
                        
                        if extracted:
                            remaining_mask_np[doc_i,sen_i] = False
                            extraction_mask_np[doc_i,sen_i] = True


            extracted_sentences = []
            extracted_sentences_positions = []

            for doc_i in range(num_documents):
                extracted_sen_indices = np.argwhere( remaining_mask_np[doc_i] == False )[:,0]
                extracted_sentences.append( [ document_batch[doc_i][sen_i] for sen_i in extracted_sen_indices if sen_i < len(document_batch[doc_i])    ] )
                extracted_sentences_positions.append( [ sen_i for sen_i in extracted_sen_indices if sen_i < len(document_batch[doc_i])  ]  )


        if return_sentence_position:
            return extracted_sentences, extracted_sentences_positions 
        else:
            return extracted_sentences




class ExtractiveSummarizer_MemSum_with_stop_sentence:
    def __init__( self, model_path, vocabulary_path, gpu = None , embed_dim=200, num_heads=8, hidden_dim = 1024, N_enc_l = 2 , N_enc_g = 2, N_dec = 3,  max_seq_len =500, max_doc_len = 100  ):
        with open( vocabulary_path , "rb" ) as f:
            words = pickle.load(f)
        self.vocab = Vocab_MemSum_with_stop_sentence( words )
        vocab_size = len(words)
        self.local_sentence_encoder = LocalSentenceEncoder_MemSum_with_stop_sentence( vocab_size, self.vocab.pad_index, embed_dim,num_heads,hidden_dim,N_enc_l, None )
        self.global_context_encoder = GlobalContextEncoder_MemSum_with_stop_sentence( embed_dim, num_heads, hidden_dim, N_enc_g )
        self.extraction_context_decoder = ExtractionContextDecoder_MemSum_with_stop_sentence( embed_dim, num_heads, hidden_dim, N_dec )
        self.extractor = Extractor_MemSum_with_stop_sentence( embed_dim, num_heads )
        ckpt = torch.load( model_path, map_location = "cpu" )
        self.local_sentence_encoder.load_state_dict( ckpt["local_sentence_encoder"] )
        self.global_context_encoder.load_state_dict( ckpt["global_context_encoder"] )
        self.extraction_context_decoder.load_state_dict( ckpt["extraction_context_decoder"] )
        self.extractor.load_state_dict(ckpt["extractor"])
        
        self.device =  torch.device( "cuda:%d"%(gpu) if gpu is not None and torch.cuda.is_available() else "cpu"  )        
        self.local_sentence_encoder.to(self.device)
        self.global_context_encoder.to(self.device)
        self.extraction_context_decoder.to(self.device)
        self.extractor.to(self.device)
        
        self.sentence_tokenizer = SentenceTokenizer_MemSum_with_stop_sentence()
        self.max_seq_len = max_seq_len
        self.max_doc_len = max_doc_len

    
    def get_ngram(self,  w_list, n = 4 ):
        ngram_set = set()
        for pos in range(len(w_list) - n + 1 ):
            ngram_set.add( "_".join( w_list[ pos:pos+n] )  )
        return ngram_set
    def extract( self, document_batch, p_stop_thres = 0.7, ngram_blocking = None, ngram = 3, return_sentence_position = False, return_sentence_score_history = False, max_extracted_sentences_per_document = 7 , stop_when_sampling_stop_sentence = True ):
        """document_batch is a batch of documents:
        [  [ sen1, sen2, ... , senL1 ], 
           [ sen1, sen2, ... , senL2], ...
         ]
        """
        ## tokenization:
        tokenized_document_batch = []
        for document in document_batch:
            tokenized_document = []
            for sen in document:
                tokenized_sen = self.sentence_tokenizer.tokenize( sen )
                tokenized_document.append( tokenized_sen )
            tokenized_document_batch.append( [self.vocab.stop_token ] + tokenized_document )


        max_document_length =  self.max_doc_len 
        max_sentence_length =  self.max_seq_len 
        ## convert to sequence
        seqs = []
        doc_mask = []
        
        for document in tokenized_document_batch:
            if len(document) > max_document_length:
                document = document[:max_document_length]
            else:
                document = document + [""] * ( max_document_length -  len(document) )
            doc_mask.append(  [ 1 if sen.strip() == "" else 0 for sen in  document   ] )
            
            document_sequences = []
            for sen in document:
                seq = self.vocab.sent2seq( sen, max_sentence_length )
                document_sequences.append(seq)
            seqs.append(document_sequences)
        seqs = np.asarray(seqs)
        doc_mask = np.asarray(doc_mask) == 1
        seqs = torch.from_numpy(seqs).to(self.device)
        doc_mask = torch.from_numpy(doc_mask).to(self.device)

        extracted_sentences = []
        sentence_score_history = []
        
        with torch.no_grad():
            num_sentences = seqs.size(1)
            sen_embed  = self.local_sentence_encoder( seqs.view(-1, seqs.size(2) )  )
            sen_embed = sen_embed.view( -1, num_sentences, sen_embed.size(1) )
            relevance_embed = self.global_context_encoder( sen_embed, doc_mask  )
    
            num_documents = seqs.size(0)
            doc_mask = doc_mask.detach().cpu().numpy()
            seqs = seqs.detach().cpu().numpy()
    
            extracted_sentences = []
            extracted_sentences_positions = []
        
            for doc_i in range(num_documents):
                current_doc_mask = doc_mask[doc_i:doc_i+1]
                current_remaining_mask_np = np.ones_like(current_doc_mask ).astype(np.bool) | current_doc_mask
                current_extraction_mask_np = np.zeros_like(current_doc_mask).astype(np.bool) | current_doc_mask
        
                current_sen_embed = sen_embed[doc_i:doc_i+1]
                current_relevance_embed = relevance_embed[ doc_i:doc_i+1 ]
                current_redundancy_embed = None
        
                current_hyps = []
                extracted_sen_ngrams = set()

                sentence_score_history_for_doc_i = []
                
                for step in range( max_extracted_sentences_per_document ) :
                    current_extraction_mask = torch.from_numpy( current_extraction_mask_np ).to(self.device)
                    current_remaining_mask = torch.from_numpy( current_remaining_mask_np ).to(self.device)
                    
                    
                    if step > 0:
                        current_redundancy_embed = self.extraction_context_decoder( current_sen_embed, current_remaining_mask, current_extraction_mask  )
                    p, _, _ = self.extractor( current_sen_embed, current_relevance_embed, current_redundancy_embed , current_extraction_mask  )
                    
            
                    p = p.masked_fill( current_extraction_mask, 1e-12 ) 

                    sentence_score_history_for_doc_i.append( p.detach().cpu().numpy()[:,1:] )

                    normalized_p = p / p.sum(dim=1, keepdims = True)

                    # stop = p_stop.squeeze(1).item()> p_stop_thres #and step > 0
                    
                    #sen_i = normalized_p.argmax(dim=1)[0]
                    sorted_sen_scores, sorted_sen_indices =normalized_p.sort(dim=1, descending= True)
                    sorted_sen_indices = sorted_sen_indices[0] 
                    sorted_sen_scores = sorted_sen_scores[0]


                    ##
                    sorted_sen_indices = sorted_sen_indices -1
                    sen_i = sorted_sen_indices.detach().cpu().numpy()[0]

                    if sen_i < 0:
                        if sorted_sen_scores[1] / sorted_sen_scores[0] > 0.15:
                            sen_i = sorted_sen_indices.detach().cpu().numpy()[:2].tolist()[-1]

                    
#                     print(sen_i)

                    if sen_i < len( document_batch[doc_i] ) :
#                         if sen_i >=0:
#                             current_hyps.append(sen_i)
                        current_hyps.append(sen_i)
                        current_extraction_mask_np[0, sen_i+1] = True
                        current_remaining_mask_np[0, sen_i+1] = False
                        
                    if stop_when_sampling_stop_sentence and sen_i < 0:
                        break

                extracted_sentences.append( [ document_batch[doc_i][sen_i] for sen_i in  current_hyps if sen_i >=0   ] )
                extracted_sentences_positions.append( [ sen_i for sen_i in  current_hyps  ]  )
                
                sentence_score_history.append(sentence_score_history_for_doc_i)

        # if return_sentence_position:
        #     return extracted_sentences, extracted_sentences_positions 
        # else:
        #     return extracted_sentences

        results = [extracted_sentences]
        if return_sentence_position:
            results.append( extracted_sentences_positions )
        if return_sentence_score_history:
            results.append( sentence_score_history )
        if len(results) == 1:
            results = results[0]
        return results
