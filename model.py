import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len    
        self.dropout = nn.Dropout(dropout)
        
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply the cos to odd positions
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        self.register_buffer('pe', pe)  # tensor

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].detach()
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # Multiplied
        self.bias = nn.Parameter(torch.zeros(1))  # Added

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # W2 and B2
        
    def forward(self, x):
        #(Batch, seq_len, d_model) --> (Batch, seq_len, d_ff) --> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        
        self.d_k = d_model // h
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    @staticmethod
    def attention(query, key, value, mask, dropout):
        #(batch, h, seq_len, d_k) -> (batch, h, seq_len, seq_len)
        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)
        attention_score = attention_score.softmax(dim=-1)
        if dropout is not None:
            attention_score = dropout(attention_score)
        #(batch, h, seq_len, seq_len) -> (batch, h, seq_len, d_k)
        return attention_score @ value, attention_score
    
    def forward(self, q, k, v, mask):
        #(batch, seq_len, d_model) -> (batch, seq_len, d_model)
        query = self.W_q(q)
        key = self.W_k(k)
        value = self.W_v(v)
        
        #(batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)  
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        
        x, _ = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        #(batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1)
        #(batch, seq_len, d_model) -> (batch, seq_len, d_model)
        return self.W_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, self_attention: MultiHeadAttentionBlock, feed_forward: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        
    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention(x, x, x, src_mask))
        x = self.residual_connection[1](x, lambda x: self.feed_forward(x))
        return x

class Encoder(nn.Module):
    def __init__(self, layer: nn.ModuleList) -> None:
        super().__init__()
        self.layer = layer
        self.norm = LayerNormalization()
        
    def forward(self, x, mask):
        for layer in self.layer:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, self_attention: MultiHeadAttentionBlock, cross_attention: MultiHeadAttentionBlock, feed_forward: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.masked_self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.masked_self_attention(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward(x))
        return x

class Decoder(nn.Module):
    def __init__(self, layer: nn.ModuleList) -> None:
        super().__init__()
        self.layer = layer
        self.norm = LayerNormalization()
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layer:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.linear = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        #(batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.linear(x), dim=-1)

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj_layer = projection_layer
    
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)   
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.proj_layer(x)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048):
    # Creating Embeddings
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)
    # Creating Positional Encoding
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Creating Encoder Blocks
    encoder_layers = nn.ModuleList([EncoderBlock(
        MultiHeadAttentionBlock(d_model, h, dropout),
        FeedForwardBlock(d_model, d_ff, dropout),
        dropout
    ) for _ in range(N)])
    
    # Creating Decoder Blocks
    decoder_layers = nn.ModuleList([DecoderBlock(
        MultiHeadAttentionBlock(d_model, h, dropout),
        MultiHeadAttentionBlock(d_model, h, dropout),
        FeedForwardBlock(d_model, d_ff, dropout),
        dropout
    ) for _ in range(N)])
    
    encoder = Encoder(encoder_layers)
    decoder = Decoder(decoder_layers)
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialising parameters for faster training
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
