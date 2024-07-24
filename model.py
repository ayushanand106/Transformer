import torch
import torch.nn as nn
import math 

class InputEmbedding(nn.Module):

    def __init(self,d_model: int,vocab_size: int):
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedding= nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):
    
    def __init__(self,d_model:int,seq_len:int,dropout:float)->None:
        super().__init__()
        self.d_model=d_model
        self.seq_len=seq_len    
        self.dropout=nn.Dropout(dropout)
        
        #Create a matrix of shape (seq_len, d_model)
        pe=torch.zeros(seq_len,d_model)
        #create a vector of shape (seq_len,1)
        position=torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
        
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        #Apply the sin to even positions
        pe[:,0::2]=torch.sin(position*div_term)
        #Apply the cos to odd positions
        pe[:,1::2]=torch.cos(position*div_term)
        pe=pe.unsqueeze(0)#(1,seq_len,d_model)
        self.register_buffer('pe',pe) # tensor

    def forward(self,x):
        x=x+(self.pe[:,x.shape[1], :]).requires_grad(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):

    def __init__(self,eps:float=10**-6)->None:
        super().__init__()
        self.eps=eps
        self.alpha=nn.Parameter(torch.ones(1)) #Multiplied
        self.bias= nn.Parameter(torch.zeros(1)) # Added

    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        std= x.std(dim=-1, keepdim=True)
        return self.alpha *(x-mean)/(std+self.eps)+ self.bias
    
     



class FeedForwardBlock(nn.Module):
    
    def __init__(self,d_model:int,d_ff:int,dropout:float)->None:
        super().__init__()
        self.linear_1=nn.Linear(d_model,d_ff)# W1 and B1
        self.dropout=nn.Dropout(dropout)
        self.linear_2=nn.Linear(d_ff,d_model)# W2 and B2
        
    def forward(self,x):
        #(Batch,seq_len,d_model)-->(Batch,seq_len,d_ff)-->(batch,seq_len,d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    


class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self,d_model:int,h:int,dropout:float)->None:
        super().__init__()
        self.d_model=d_model
        self.h=h
        assert d_model%h==0,"d_model is not divisible by h"
        
        self.d_k=d_model/h
        self.W_q=nn.Linear(d_model,d_model)
        self.W_k=nn.Linear(d_model,d_model) 
        self.W_v=nn.Linear(d_model,d_model)
        self.W_o=nn.Linear(d_model,d_model)
        
        self.dropout=nn.Dropout(dropout)
        
        
    @staticmethod # this makes function to be called without creating instance of class
    def attention(self,query,key,value,mask,dropout: nn.Dropout):
        #(batch,h,seq_len,d_k)->(batch,h,seq_len,seq_len)
        attention_score=(query@key.transpose(-2,-1))/math.sqrt(self.d_k)
        if mask is not None:
            attention_score.mask_filled(mask==0,-1e9)
        attention_score=attention_score.softmax(dim=-1)
        if dropout is not None:
            attention_score=dropout(attention_score)    
        #(batch,h,seq_len,seq,len)-->(batch,h,seq_len,d_k)
        return (attention_score@value,attention_score)
        
    
    def forward(self,q,k,v,mask):
        #(batch,seq_len,d_model)->(batch,seq_len,d_model)
        query=self.W_q(q) 
        key=self.W_k(k)
        value=self.W_v(v)
        
        #(batch,seq_len,d_model)->(batch,seq_len,h,d_k)->(batch,h,seq_len,d_k)  
        query=query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        value=value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)
        key=key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        x=MultiHeadAttentionBlock.attention(query=query,value=value,dropout=self.dropout)
        #(batch,h,seq_len,d_k)->(batch,seq_len,h,d_k)->(batch,seq_len,d_model)
        x=x.transpose(1,2).contiguous().view(x.shape[0],x.shape[1],-1,self.h*self.d_k)
        #(batch,seq_len,d_model)->(batch,seq_len,d_model)
        return (self.W_o(x))
        
        
class ResidualConnection(nn.Module):
    def __init__(self,dropout:float) -> None:
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.norm=LayerNormalization()
        
    def forward(self,x,sublayer):
        return x+ self.dropout(sublayer(self.norm(x)))
    
    
class EncoderBlock(nn.Module):
    def __init__(self, self_attention:MultiHeadAttentionBlock, feed_forward:FeedForwardBlock,dropout:float) -> None:
        super().__init__()
        self.self_attention=self_attention
        self.feed_forward=feed_forward
        self.residual_connection=nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        
    def forward(self,x,src_mask):
        x=self.residual_connection[0](x, lambda X:self.self_attention(x,x,x,src_mask))
        x=self.residual_connection[1](x, lambda X:self.feed_forward(x))
        return x
    

class Encoder(nn.Module):
    def __init__(self,layer:nn.ModuleList) -> None:
        super().__init__()
        self.layer=layer
        self.norm=LayerNormalization()
        
    def forward(self,x,mask):
        for layer in self.layer:
            x=layer(x,mask)
        return self.norm(x)
        
        
class DecoderBlock(nn.Module):
    def __init__(self,self_attention:MultiHeadAttentionBlock,cross_attention:MultiHeadAttentionBlock,feed_forward,dropout:float) -> None:
        super().__init__()
        self.masked_self_attention=self_attention
        self.cross_attention=cross_attention
        self.feed_forward=feed_forward
        self.residual_connections=nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
        
    def forward(self,x, encoder_output, src_mask ,tgt_mask):
        x=self.residual_connections[0](x,lambda x:self.masked_self_attention(x,x,x,tgt_mask))
        x=self.residual_connections[1](x,lambda x:self.cross_attention(x,encoder_output,encoder_output,src_mask))
        x=self.residual_connections[2](x,self.feed_forward(x))
        return x
    
class Decoder(nn.Module):
    def __init__(self,layer:nn.ModuleList) -> None:
        super().__init__()
        self.layer=layer
        self.norm=LayerNormalization()
        
    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for layer in self.layer:
            x=layer(x,encoder_output,src_mask,tgt_mask)
        return self.norm(x)
        
class Projection_layer(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.linear=nn.Linear(d_model,vocab_size)
    
    def forward(self,x):
        #(batch,seq_len,d_model)->(batch,seq_len,vocab_size)
        return torch.log_softmax(self.linear(x),dim=-1)
    
class Transformer(nn.Module):
    def __init__(self, encoder:Encoder ,decoder:Decoder, src_embed:InputEmbedding , tgt_embed:InputEmbedding,src_pos:PositionalEncoding,tgt_pos:PositionalEncoding,projection_layer:Projection_layer) -> None:
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.tgt_embed=tgt_embed
        self.src_embed=src_embed
        self.src_pos=src_pos
        self.tgt_embed=tgt_pos
        self.proj_layer=projection_layer
    
    def encode(self,src,src_mask):
        src=self.src_embed(src)
        src=self.src_pos(src)
        return self.encoder(src,src_mask)
    
    def decode(self,encoder_output,src_mask,tgt,tgt_mask):
        tgt=self.tgt_embed(tgt)
        tgt=self.tgt_pos(tgt)   
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)
    
    def project(self,x):
        return self.proj_layer(x)
    
    

def build_transformer(self,src_vocab_size:int,tgt_vocab_size:int,src_seq_len:int,tgt_seq_len:int,d_model:int=512, N:int=6, h:int=8,dropout:int=0.1,d_ff=2048):
    #Creating Embeddings
    src_embed=InputEmbedding(d_model,src_vocab_size)
    tgt_embed=InputEmbedding(d_model,tgt_vocab_size)
    #Creating positional Encoding
    src_pos=PositionalEncoding(d_model,src_seq_len,dropout)
    tgt_pos=PositionalEncoding(d_model,tgt_seq_len,dropout)
    
    #Creating encoder blocks
    encoder_blocks=[]
    for _ in range(N):
        encoder_self_attention=MultiHeadAttentionBlock(d_model,h,dropout)
        encoder_feed_forward=FeedForwardBlock(d_model,d_ff,dropout)
        encoder_block=EncoderBlock(encoder_self_attention,encoder_feed_forward,dropout)
        encoder_blocks.append(encoder_block)
    
    #Creating decoder blocks
    decoder_blocks=[]
    for _ in range(N):
        decoder_self_attention=MultiHeadAttentionBlock(d_model,h,dropout)
        decoder_cross_attention=MultiHeadAttentionBlock(d_model,h,dropout)
        decoder_feed_forward=FeedForwardBlock(d_model,d_ff,dropout)
        decoder_block=DecoderBlock(decoder_self_attention,decoder_cross_attention,decoder_feed_forward,dropout)
        decoder_blocks.append(decoder_block)
    
    #creating encoder
    encoder=Encoder(nn.ModuleList(encoder_blocks))
    #creating decoder
    decoder=Decoder(nn.ModuleList(decoder_blocks))
    
    proj_layer=Projection_layer(d_model,tgt_vocab_size)
    transformer=Transformer(encoder,decoder,src_embed,tgt_embed,src_pos,tgt_pos,proj_layer)
    
    #Initialising parameters for faster training
    for p in transformer.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
            
    return transformer