import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.distributions import Categorical

class AddMask( nn.Module ):
    def __init__( self, pad_index ):
        super().__init__()
        self.pad_index = pad_index
    def forward( self, x):
        # here x is a batch of input sequences (not embeddings) with the shape of [ batch_size, seq_len]
        mask = x == self.pad_index
        return mask


class PositionalEncoding( nn.Module ):
    def __init__(self,  embed_dim, max_seq_len = 512  ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        pe = torch.zeros( 1, max_seq_len,  embed_dim )
        for pos in range( max_seq_len ):
            for i in range( 0, embed_dim, 2 ):
                pe[ 0, pos, i ] = math.sin( pos / ( 10000 ** ( i/embed_dim ) )  )
                if i+1 < embed_dim:
                    pe[ 0, pos, i+1 ] = math.cos( pos / ( 10000** ( i/embed_dim ) ) )
        self.register_buffer( "pe", pe )
        ## register_buffer can register some variables that can be saved and loaded by state_dict, but not trainable since not accessible by model.parameters()
    def forward( self, x ):
        return x + self.pe[ :, : x.size(1), :]



class MultiHeadAttention( nn.Module ):
    def __init__(self, embed_dim, num_heads ):
        super().__init__()
        dim_per_head = int( embed_dim/num_heads )
        
        self.ln_q = nn.Linear( embed_dim, num_heads * dim_per_head )
        self.ln_k = nn.Linear( embed_dim, num_heads * dim_per_head )
        self.ln_v = nn.Linear( embed_dim, num_heads * dim_per_head )

        self.ln_out = nn.Linear( num_heads * dim_per_head, embed_dim )

        self.num_heads = num_heads
        self.dim_per_head = dim_per_head
    
    def forward( self, q,k,v, mask = None):
        q = self.ln_q( q )
        k = self.ln_k( k )
        v = self.ln_v( v )

        q = q.view( q.size(0), q.size(1),  self.num_heads, self.dim_per_head  ).transpose( 1,2 )
        k = k.view( k.size(0), k.size(1),  self.num_heads, self.dim_per_head  ).transpose( 1,2 )
        v = v.view( v.size(0), v.size(1),  self.num_heads, self.dim_per_head  ).transpose( 1,2 )

        a = self.scaled_dot_product_attention( q,k, mask )
        new_v = a.matmul(v)
        new_v = new_v.transpose( 1,2 ).contiguous()
        new_v = new_v.view( new_v.size(0), new_v.size(1), -1 )
        new_v = self.ln_out(new_v)
        return new_v

    def scaled_dot_product_attention( self, q, k, mask = None ):
        ## note the here q and k have converted into multi-head mode 
        ## q's shape is [ Batchsize, num_heads, seq_len_q, dim_per_head ]
        ## k's shape is [ Batchsize, num_heads, seq_len_k, dim_per_head ]
        # scaled dot product
        a = q.matmul( k.transpose( 2,3 ) )/ math.sqrt( q.size(-1) )
        # apply mask (either padding mask or seqeunce mask)
        if mask is not None:
            a = a.masked_fill( mask.unsqueeze(1).unsqueeze(1) , -1e9 )  
        # apply softmax, to get the likelihood as attention matrix
        a = F.softmax( a, dim=-1 )
        return a

class FeedForward( nn.Module ):
    def __init__( self, embed_dim, hidden_dim ):
        super().__init__()
        self.ln1 = nn.Linear( embed_dim, hidden_dim )
        self.ln2 = nn.Linear( hidden_dim, embed_dim )
    def forward(  self, x):
        net = F.relu(self.ln1(x))
        out = self.ln2(net)
        return out


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim ):
        super().__init__()
        self.mha = MultiHeadAttention( embed_dim, num_heads  )
        self.norm1 = nn.LayerNorm( embed_dim )
        self.feed_forward = FeedForward( embed_dim, hidden_dim )
        self.norm2 = nn.LayerNorm( embed_dim )
    def forward( self, x, mask, dropout_rate = 0. ):
        short_cut = x
        net = F.dropout(self.mha( x,x,x, mask ), p = dropout_rate)
        net = self.norm1( short_cut + net )
        short_cut = net
        net = F.dropout(self.feed_forward( net ), p = dropout_rate )
        net = self.norm2( short_cut + net )
        return net

class TransformerDecoderLayer( nn.Module ):
    def __init__(self, embed_dim, num_heads, hidden_dim ):
        super().__init__()
        self.masked_mha = MultiHeadAttention(  embed_dim, num_heads )
        self.norm1 = nn.LayerNorm( embed_dim )
        self.mha = MultiHeadAttention( embed_dim, num_heads )
        self.norm2 = nn.LayerNorm( embed_dim )
        self.feed_forward = FeedForward( embed_dim, hidden_dim )
        self.norm3 = nn.LayerNorm( embed_dim )
    def forward(self, encoder_output, x, src_mask, trg_mask , dropout_rate = 0. ):
        short_cut = x
        net = F.dropout(self.masked_mha( x,x,x, trg_mask ), p = dropout_rate)
        net = self.norm1( short_cut + net )
        short_cut = net
        net = F.dropout(self.mha( net, encoder_output, encoder_output, src_mask ), p = dropout_rate)
        net = self.norm2( short_cut + net )
        short_cut = net
        net = F.dropout(self.feed_forward( net ), p = dropout_rate)
        net = self.norm3( short_cut + net )
        return net 

class MultiHeadPoolingLayer( nn.Module ):
    def __init__( self, embed_dim, num_heads  ):
        super().__init__()
        self.num_heads = num_heads
        self.dim_per_head = int( embed_dim/num_heads )
        self.ln_attention_score = nn.Linear( embed_dim, num_heads )
        self.ln_value = nn.Linear( embed_dim,  num_heads * self.dim_per_head )
        self.ln_out = nn.Linear( num_heads * self.dim_per_head , embed_dim )
    def forward(self, input_embedding , mask=None):
        a = self.ln_attention_score( input_embedding )
        v = self.ln_value( input_embedding )
        
        a = a.view( a.size(0), a.size(1), self.num_heads, 1 ).transpose(1,2)
        v = v.view( v.size(0), v.size(1),  self.num_heads, self.dim_per_head  ).transpose(1,2)
        a = a.transpose(2,3)
        if mask is not None:
            a = a.masked_fill( mask.unsqueeze(1).unsqueeze(1) , -1e9 ) 
        a = F.softmax(a , dim = -1 )

        new_v = a.matmul(v)
        new_v = new_v.transpose( 1,2 ).contiguous()
        new_v = new_v.view( new_v.size(0), new_v.size(1) ,-1 ).squeeze(1)
        new_v = self.ln_out( new_v )
        return new_v


# class LocalSentenceEncoder( nn.Module ):
#     def __init__( self, vocab_size, pad_index, embed_dim, num_heads , hidden_dim , num_enc_layers , pretrained_word_embedding ):
#         super().__init__()
#         self.addmask = AddMask( pad_index )
#         self.pos_encode = PositionalEncoding( embed_dim)
#         self.layer_list = nn.ModuleList([ TransformerEncoderLayer( embed_dim, num_heads, hidden_dim ) for _ in range(num_enc_layers) ])
#         self.mh_pool = MultiHeadPoolingLayer( embed_dim, num_heads )

#         if pretrained_word_embedding is not None:
#             ## make sure the pad embedding is 0
#             pretrained_word_embedding[pad_index] = 0
#             self.register_buffer( "word_embedding", torch.from_numpy( pretrained_word_embedding ) )
#         else:
#             self.register_buffer( "word_embedding", torch.randn( vocab_size, embed_dim ) )

#     """
#     input_seq 's shape:  batch_size x seq_len 
#     """
#     def forward( self, input_seq, dropout_rate = 0. ):
#         mask = self.addmask( input_seq )
#         ## batch_size x seq_len x embed_dim
#         net = self.word_embedding[ input_seq ]
#         net = self.pos_encode( net )
#         for layer in self.layer_list:
#             net = layer( net, mask, dropout_rate )
#         net = self.mh_pool( net, mask )
#         return net


class LocalSentenceEncoder( nn.Module ):
    def __init__( self, vocab_size, pad_index, embed_dim, num_heads , hidden_dim , num_enc_layers , pretrained_word_embedding ):
        super().__init__()
        self.addmask = AddMask( pad_index )
      
        self.rnn = nn.LSTM(  embed_dim, embed_dim, 2, batch_first = True, bidirectional = True)
        self.mh_pool = MultiHeadPoolingLayer( 2*embed_dim, num_heads )
        self.norm_out = nn.LayerNorm( 2*embed_dim )
        self.ln_out = nn.Linear( 2*embed_dim, embed_dim )

        if pretrained_word_embedding is not None:
            ## make sure the pad embedding is 0
            pretrained_word_embedding[pad_index] = 0
            self.register_buffer( "word_embedding", torch.from_numpy( pretrained_word_embedding ) )
        else:
            self.register_buffer( "word_embedding", torch.randn( vocab_size, embed_dim ) )

    """
    input_seq 's shape:  batch_size x seq_len 
    """
    def forward( self, input_seq, dropout_rate = 0. ):
        mask = self.addmask( input_seq )
        ## batch_size x seq_len x embed_dim
        net = self.word_embedding[ input_seq ]
        net, _ = self.rnn( net )
        net =  self.ln_out(F.relu(self.norm_out(self.mh_pool( net, mask ))))
        return net


# class GlobalContextEncoder(nn.Module):
#     def __init__(self, embed_dim,  num_heads, hidden_dim, num_dec_layers ):
#         super().__init__()
#         self.pos_encode = PositionalEncoding( embed_dim)
#         self.layer_list = nn.ModuleList( [  TransformerEncoderLayer( embed_dim, num_heads, hidden_dim ) for _ in range(num_dec_layers) ] )

#     def forward(self, sen_embed, doc_mask, dropout_rate = 0.):
#         net = self.pos_encode( sen_embed )
#         for layer in self.layer_list:
#             net = layer( net, doc_mask, dropout_rate )
#         sen_context_embed = net
#         return sen_context_embed


class GlobalContextEncoder(nn.Module):
    def __init__(self, embed_dim,  num_heads, hidden_dim, num_dec_layers ):
        super().__init__()
        # self.pos_encode = PositionalEncoding( embed_dim)
        # self.layer_list = nn.ModuleList( [  TransformerEncoderLayer( embed_dim, num_heads, hidden_dim ) for _ in range(num_dec_layers) ] )
        self.rnn = nn.LSTM(  embed_dim, embed_dim, 2, batch_first = True, bidirectional = True)
        self.norm_out = nn.LayerNorm( 2*embed_dim )
        self.ln_out = nn.Linear( 2*embed_dim, embed_dim )

    def forward(self, sen_embed, doc_mask, dropout_rate = 0.):
        net, _ = self.rnn( sen_embed )
        net = self.ln_out(F.relu( self.norm_out(net) ) )
        return net


class ExtractionContextDecoder( nn.Module ):
    def __init__( self, embed_dim,  num_heads, hidden_dim, num_dec_layers ):
        super().__init__()
        self.layer_list = nn.ModuleList( [  TransformerDecoderLayer( embed_dim, num_heads, hidden_dim ) for _ in range(num_dec_layers) ] )
    ## remaining_mask: set all unextracted sen indices as True
    ## extraction_mask: set all extracted sen indices as True
    def forward( self, sen_embed, remaining_mask, extraction_mask, dropout_rate = 0. ):
        net = sen_embed
        for layer in self.layer_list:
            #  encoder_output, x,  src_mask, trg_mask , dropout_rate = 0.
            net = layer( sen_embed, net, remaining_mask, extraction_mask, dropout_rate )
        return net

class Extractor( nn.Module ):
    def __init__( self, embed_dim, num_heads ):
        super().__init__()
        self.norm_input = nn.LayerNorm( 3*embed_dim  )
        
        self.ln_hidden1 = nn.Linear(  3*embed_dim, 2*embed_dim  )
        self.norm_hidden1 = nn.LayerNorm( 2*embed_dim  )
        
        self.ln_hidden2 = nn.Linear(  2*embed_dim, embed_dim  )
        self.norm_hidden2 = nn.LayerNorm( embed_dim  )

        self.ln_out = nn.Linear(  embed_dim, 1 )

        self.mh_pool = MultiHeadPoolingLayer( embed_dim, num_heads )
        self.norm_pool = nn.LayerNorm( embed_dim  )
        self.ln_stop = nn.Linear(  embed_dim, 1 )

        self.mh_pool_2 = MultiHeadPoolingLayer( embed_dim, num_heads )
        self.norm_pool_2 = nn.LayerNorm( embed_dim  )
        self.ln_baseline = nn.Linear(  embed_dim, 1 )

    def forward( self, sen_embed, relevance_embed, redundancy_embed , extraction_mask, dropout_rate = 0. ):
        if redundancy_embed is None:
            redundancy_embed = torch.zeros_like( sen_embed )
        net = self.norm_input( F.dropout( torch.cat( [ sen_embed, relevance_embed, redundancy_embed ], dim = 2 ) , p = dropout_rate  )  ) 
        net = F.relu( self.norm_hidden1( F.dropout( self.ln_hidden1( net ) , p = dropout_rate  )   ))
        hidden_net = F.relu( self.norm_hidden2( F.dropout( self.ln_hidden2( net)  , p = dropout_rate  )  ))
        
        p = self.ln_out( hidden_net ).sigmoid().squeeze(2)

        net = F.relu( self.norm_pool(  F.dropout( self.mh_pool( hidden_net, extraction_mask) , p = dropout_rate  )  ))
        p_stop = self.ln_stop( net ).sigmoid().squeeze(1)

        net = F.relu( self.norm_pool_2(  F.dropout( self.mh_pool_2( hidden_net, extraction_mask ) , p = dropout_rate  )  ))
        baseline = self.ln_baseline(net)

        return p, p_stop, baseline