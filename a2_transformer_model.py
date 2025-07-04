"""
This code is provided solely for the personal and private use of students
taking the CSC401H/2511H course at the University of Toronto. Copying for
purposes other than this use is expressly prohibited. All forms of
distribution of this code, including but not limited to public repositories on
GitHub, GitLab, Bitbucket, or any other online platform, whether as given or
with any changes, are expressly prohibited.

Author: Arvid Frydenlund <arvie@cs.toronto.edu>

All of the files in this directory and all subdirectories are:
Copyright (c) 2024 University of Toronto
"""

import math
import torch
import torch.nn as nn

from torch import Tensor
from typing import Tuple

# import a2_utils


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x: Tensor):
        """
        Compute layer normalization
            y = gamma * (x - mu) / (sigma + eps) + beta where mu and sigma are computed over the feature dimension

        x: torch.Tensor, shape [batch_size, seq_len, d_model]
        return: torch.Tensor, shape [batch_size, seq_len, d_model]
        """
        mu = x.mean(keepdim=True, dim=-1)
        sigma = x.std(dim = -1, keepdim = True)
        
        gamma = self.gamma.view(1, 1, -1)  
        beta = self.beta.view(1, 1, -1)    
        return (beta + (gamma * (x - mu)) / (sigma + self.eps))

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention for both self-attention and cross-attention
    """

    def __init__(
        self,
        num_heads: int,
        d_model: int,
        dropout: float = 0.0,
        atten_dropout: float = 0.0,
        store_attention_scores: bool = False,
    ):
        """
        num_heads: int, the number of heads
        d_model: int, the dimension of the model
        dropout: float, the dropout rate
        atten_dropout: float, the dropout rate for the attention i.e. drops out full tokens
        store_attention_scores: bool, whether to store the attention scores for visualization
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        # Assume values and keys are the same size
        self.d_head = d_model // num_heads
        self.num_heads = num_heads

        # Note for students, for self-attention, it is more efficient to treat q, k, and v as one matrix
        # but this way allows us to use the same attention function for cross-attention
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.atten_dropout = nn.Dropout(p=atten_dropout)  # applied after softmax

        # applied at the end
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

        # used for visualization
        self.store_attention_scores = store_attention_scores
        self.attention_scores = None  # set by set_attention_scores

    def set_attention_scores(self, scores: Tensor) -> None:
        """
        A helper function for visualization of attention scores.
        These are stored as attributes so that students do not need to deal with passing them around.

        The attention scores should be given after masking but before the softmax.
        scores: torch.Tensor, shape [batch_size, num_heads, query_seq_len, key_seq_len]
        return: None
        """
        if scores is None:  # for clean up
            self.attention_scores = None
        if self.store_attention_scores and not self.training:
            self.attention_scores = scores.cpu().detach().numpy()

    def attention(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        """
        Scaled dot product attention
        Hint: the mask is applied before the softmax.
        Hint: attention dropout `self.atten_dropout` is applied to the attention weights after the softmax.

        You are required to make comments about the shapes of the tensors at each step of the way
        in order to assist the markers.  Does a tensor change shape?  Make a comment.

        You are required to call set_attention_scores with the correct tensor before returning from this function.
        The attention scores should be given after masking but before the softmax.
        This is for testing purposes and potentially for other uses.

        query: torch.Tensor, shape [batch_size, num_heads, query_seq_len, d_head]
        key: torch.Tensor, shape [batch_size, num_heads, key_seq_len, d_head]
        value: torch.Tensor, shape [batch_size, num_heads, key_seq_len, d_head]
        mask:  torch.Tensor, shape [batch_size, query_seq_len, key_seq_len,], True, where masked or None

        return torch.Tensor, shape [batch_size, num_heads, query_seq_len, d_head]
        """
        d_head = torch.tensor(self.d_head, dtype= torch.float32)
        attention_score = torch.matmul(query, key.transpose(-2, -1))/ torch.sqrt(d_head)
        # Attention score changes the shape of the tensor to 
        # [batch_size, num_heads, query_seq_len, key_seq_len]
        if mask is not None:
            mask = mask.to(attention_score.device)
            mask = mask.unsqueeze(1)
            attention_score = attention_score.masked_fill(mask, -1e9)
        self.set_attention_scores(attention_score)
        weights = torch.softmax(attention_score, dim= -1)
        weights = self.atten_dropout(weights)
        return torch.matmul(weights, value)
        # The final matmul also changes the shape of the tensor to the desired output shape 
        # [batch_size, num_heads, query_seq_len, d_head]

    def forward(self, query: Tensor, key: Tensor = None, value: Tensor = None, mask: Tensor = None) -> Tensor:
        """
        If the key and values are None, assume self-attention is being applied.  Otherwise, assume cross-attention.

        Note we only need one mask, which will work for either causal self-attention or cross-attention as long as
        the mask is set up properly beforehand.

        You are required to make comments about the shapes of the tensors at each step of the way
        in order to assist the markers.  Does a tensor change shape?  Make a comment.

        query: torch.Tensor, shape [batch_size, query_seq_len, d_model]
        key: torch.Tensor, shape [batch_size, key_seq_len, d_model] or None
        value: torch.Tensor, shape [batch_size, key_seq_len, d_model] or None
        mask: torch.Tensor, shape [batch_size, query_seq_len, key_seq_len,], True where masked or None

        return: torch.Tensor, shape [batch_size, query_seq_len, d_model]
        """
        if key is None and value is None:
            key = query
            value = query

        batch_size = query.size(0)
        query = self.q_linear(query)
        key = self.k_linear(key) 
        value = self.v_linear(value)

        # Here all three of query key and value tensors change in shape.
        # It changes to [batch_size, num_heads, seq_len, d_head] from 
        # [batch_size, seq_len, d_model]
        query = query.reshape(batch_size, -1, self.num_heads, 
                              self.d_head).transpose(1,2)
        key = key.reshape(batch_size, -1, self.num_heads, 
                              self.d_head).transpose(1,2)
        value = value.reshape(batch_size, -1, self.num_heads, 
                              self.d_head).transpose(1,2)

        scaled_attention = self.attention(query, key, value, mask)
        # Here below we change the shape of the tensor from 
        # [batch_size, num_heads, seq_len, d_head] to 
        # [batch_size, seq_len, d_model]
        scaled_attention = scaled_attention.transpose(1,2).reshape(batch_size, 
                                            -1, self.num_heads * self.d_head)
        
        scaled_attention = self.out_linear(scaled_attention)
        return self.dropout(scaled_attention)
        
class FeedForwardLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(FeedForwardLayer, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.f = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the feedforward sublayer.
        Dropout is applied after the activation function and after the second linear layer

        x: torch.Tensor, shape [batch_size, seq_len, d_model]
        return: torch.Tensor, shape [batch_size, seq_len, d_model]
        """
        x = self.w_1(x)
        activation_1 = self.f(x)
        x = self.dropout1(activation_1)
        activation_2 = self.w_2(x)  
        x = self.dropout2(activation_2)
        return x

class TransformerEncoderLayer(nn.Module):
    """

    Idea if we can give this init done, then the students can fill in the decoder init in the same way but add in cross attention


    Performs multi-head self attention and FFN with the desired pre- or post-layer norm and residual connections.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout: float = 0.1,
        atten_dropout: float = 0.0,
        is_pre_layer_norm: bool = True,
    ):
        """
        d_model: int, the dimension of the model
        d_ff: int, the dimension of the feedforward network interior projection
        num_heads: int, the number of heads for the multi-head attention
        dropout: float, the dropout rate
        atten_dropout: float, the dropout rate for the attention i.e. drops out full tokens
        """
        super(TransformerEncoderLayer, self).__init__()
        self.is_pre_layer_norm = is_pre_layer_norm
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_head = num_heads

        self.ln1 = LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(
            num_heads, d_model, dropout=dropout, atten_dropout=atten_dropout
        )

        self.ln2 = LayerNorm(d_model)
        self.ff = FeedForwardLayer(d_model, d_ff, dropout=dropout)

    def pre_layer_norm_forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        x: torch.Tensor, the input to the layer
        mask: torch.Tensor, the mask to apply to the attention
        Hint:  should only require two or three lines of code
        """
        
        normal_1 = self.ln1(x)
        x = x + self.self_attn(normal_1, normal_1, normal_1, mask) 
        normal_2 = self.ln2(x)
        return x + self.ff(normal_2)

    def post_layer_norm_forward(self, x: Tensor, mask: Tensor) -> Tensor:
        attention_values = self.self_attn(x, x, x, mask)
        x = self.ln1(x + attention_values)
        return self.ln2(x + self.ff(x))

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        if self.is_pre_layer_norm:
            return self.pre_layer_norm_forward(x, mask)
        else:
            return self.post_layer_norm_forward(x, mask)

class TransformerEncoder(nn.Module):
    """
    Stacks num_layers of TransformerEncoderLayer and applies layer norm at the correct place.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout: float = 0.1,
        atten_dropout: float = 0.0,
        is_pre_layer_norm: bool = True,
    ):
        super(TransformerEncoder, self).__init__()
        self.is_pre_layer_norm = is_pre_layer_norm
        self.layers = torch.nn.ModuleList()
        for l in range(num_layers):
            self.layers.append(
                TransformerEncoderLayer(
                    d_model, d_ff, num_heads, dropout, atten_dropout, is_pre_layer_norm
                )
            )
        self.norm = LayerNorm(d_model)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        x: torch.Tensor, the input to the encoder
        mask: torch.Tensor, the mask to apply to the attention
        """
        if not self.is_pre_layer_norm:
            x = self.norm(x)
        for layer in self.layers:
            x = layer(x, mask)
        if self.is_pre_layer_norm:
            x = self.norm(x)
        return x

class TransformerDecoderLayer(nn.Module):
    """
    Performs multi-head self attention, multi-head cross attention, and FFN,
    with the desired pre- or post-layer norm and residual connections.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout: float = 0.1,
        atten_dropout: float = 0.0,
        is_pre_layer_norm: bool = True,
    ):
        """
        d_model: int, the dimension of the model
        d_ff: int, the dimension of the feedforward network interior projection
        num_heads: int, the number of heads for the multi-head attention
        dropout: float, the dropout rate
        atten_dropout: float, the dropout rate for the attention i.e. drops out full tokens
        is_pre_layer_norm: bool, whether to apply layer norm before or after each sublayer

        Please use the following attribute names 'self_attn', 'cross_attn', and 'ff' and any others you think you need.
        """
        super(TransformerDecoderLayer, self).__init__()
        self.is_pre_layer_norm = is_pre_layer_norm
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_head = num_heads
           
        self.ln1 = LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(
            num_heads, d_model, dropout=dropout, atten_dropout=atten_dropout
        )
        self.cross_attn = MultiHeadAttention(
            num_heads, d_model, dropout=dropout, atten_dropout=atten_dropout
        )
        self.ln2 = LayerNorm(d_model)
        self.ln3 = LayerNorm(d_model)
        self.ff = FeedForwardLayer(d_model, d_ff, dropout=dropout)

    def pre_layer_norm_forward(self, x: Tensor, mask: Tensor, src_x: Tensor, src_mask: Tensor) -> Tensor:
        normal_1 = self.ln1(x)
        x = x + self.self_attn(normal_1, normal_1, normal_1, mask)
        normal_2 = self.ln2(x)
        normal_2_cross = self.ln2(src_x)
        x = x + self.cross_attn(normal_2, normal_2_cross, normal_2_cross, src_mask)
        normal_3 = self.ln3(x)
        return x + self.ff(normal_3)


    def post_layer_norm_forward(self, x: Tensor, mask: Tensor, src_x: Tensor, src_mask: Tensor) -> Tensor:
        x = self.ln1(x + self.self_attn(x, x, x, mask))
        x = self.ln2(x + self.cross_attn(x, src_x, src_x, src_mask))
        return self.ln3(x + self.ff(x))

    def forward(self, x: Tensor, mask: Tensor, src_x: Tensor, src_mask: Tensor) -> Tensor:
        if self.is_pre_layer_norm:
            return self.pre_layer_norm_forward(x, mask, src_x, src_mask)
        else:
            return self.post_layer_norm_forward(x, mask, src_x, src_mask)

    def store_attention_scores(self, should_store: bool = True) -> None:
        self.self_attn.store_attention_scores = should_store
        self.cross_attn.store_attention_scores = should_store

    def get_attention_scores(self):
        return self.self_attn.attention_scores, self.cross_attn.attention_scores

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout: float = 0.1,
        atten_dropout: float = 0.0,
        is_pre_layer_norm: bool = True,
    ):
        super(TransformerDecoder, self).__init__()
        self.is_pre_layer_norm = is_pre_layer_norm
        self.layers = torch.nn.ModuleList()
        for l in range(num_layers):
            self.layers.append(
                TransformerDecoderLayer(
                    d_model, d_ff, num_heads, dropout, atten_dropout, is_pre_layer_norm
                )
            )
        self.norm = LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)  # logit projection

    def forward(self, x: Tensor, mask: Tensor, src_x: Tensor, src_mask: Tensor,
                normalize_logits: bool = False) -> Tensor:
        """
        x: torch.Tensor, the input to the decoder
        mask: torch.Tensor, the mask to apply to the attention
        src_x: torch.Tensor, the output of the encoder
        src_mask: torch.Tensor, the mask to apply to the attention
        normalize_logits: bool, whether to apply log_softmax to the logits

        Returns the logits or log probabilities if normalize_logits is True

        Hint: look at the encoder for how pre/post layer norm is handled
        """
        if not self.is_pre_layer_norm:
            x = self.norm(x)
        for layer in self.layers:
            x = layer(x, mask, src_x, src_mask)
        if self.is_pre_layer_norm:
            x = self.norm(x)
        projection = self.proj(x)
        if normalize_logits:
            return torch.log_softmax(projection, dim = -1)
        else:
            return projection
            

    def store_attention_scores(self, should_store: bool = True) -> None:
        for layer in self.layers:
            layer.store_attention_scores(should_store)

    def get_attention_scores(self):
        """
        Return the attention scores (self-attention, cross-attention) from all layers
        """
        scores = []
        for layer in self.layers:
            scores.append(layer.get_attention_scores())
        return scores

class TransformerEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super(TransformerEmbeddings, self).__init__()
        self.lookup = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: Tensor) -> Tensor:
        """
        x: torch.Tensor, shape [batch_size, seq_len] of int64 in range [0, vocab_size)
        return torch.Tensor, shape [batch_size, seq_len, d_model]
        """
        return self.lookup(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        pos = self.pe[:, : x.size(1)].requires_grad_(False)
        x = x + pos  # Add the position encoding to original vector x
        return self.dropout(x)

class TransformerEncoderDecoder(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        padding_idx: int,
        num_layers: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout: float = 0.1,
        atten_dropout: float = 0.0,
        is_pre_layer_norm: bool = True,
        no_src_pos: bool = False,
        no_tgt_pos: bool = False,
    ):
        super(TransformerEncoderDecoder, self).__init__()
        """
        src_vocab_size: int, the size of the source vocabulary
        tgt_vocab_size: int, the size of the target vocabulary
        padding_idx: int, the index of the pad token
        num_layers: int, the number of layers
        d_model: int, the dimension of the model
        d_ff: int, the dimension of the feedforward network interior projection
        num_heads: int, the number of heads for the multi-head attention
        dropout: float, the dropout rate
        atten_dropout: float, the dropout rate for the attention i.e. drops out full tokens
        is_pre_layer_norm: bool, whether to apply layer norm before or after each sublayer
        no_src_pos: bool, whether to skip positional encoding for the source
        no_tgt_pos: bool, whether to skip positional encoding for the target
        """

        self.src_embed = TransformerEmbeddings(src_vocab_size, d_model)
        if no_src_pos:
            self.src_pe = None
            print("Warning: no positional encoding for the source")
        else:
            self.src_pe = PositionalEncoding(d_model, dropout)
        self.tgt_embed = TransformerEmbeddings(tgt_vocab_size, d_model)
        if no_tgt_pos:
            self.tgt_pe = None
            print("Warning: no positional encoding for the target")
        else:
            self.tgt_pe = PositionalEncoding(d_model, dropout)

        self.encoder = TransformerEncoder(
            num_layers,
            d_model,
            d_ff,
            num_heads,
            dropout,
            atten_dropout,
            is_pre_layer_norm,
        )
        self.decoder = TransformerDecoder(
            tgt_vocab_size,
            num_layers,
            d_model,
            d_ff,
            num_heads,
            dropout,
            atten_dropout,
            is_pre_layer_norm,
        )

        self.padding_idx = padding_idx

    def create_pad_mask(self, tokens: Tensor) -> Tensor:
        """
        Create a padding mask using pad_idx (an attribute of the class)
        Hint: respect the output shape

        tokens: torch.Tensor, [batch_size, seq_len]
        return: torch.Tensor, [batch_size, 1, seq_len] where True means to mask, and on the same device as tokens
        """
        padding_mask = tokens == self.padding_idx
        return padding_mask.unsqueeze(1).to(tokens.device) 

    @staticmethod
    def create_causal_mask(tokens: Tensor) -> Tensor:
        """
        Create a causal (upper) triangular mask
        Hint: respect the output shape and this can be done via torch.triu

        tokens: torch.Tensor, [batch_size, seq_len]
        pad_idx: int, the index of the pad token
        return: torch.Tensor, [1, seq_len, seq_len] where True means to mask, and on the same device as tokens
            Hint, if seq_len = 5, then the mask should look like:
                tensor([[[False,  True,  True,  True,  True],
                         [False, False,  True,  True,  True],
                         [False, False, False,  True,  True],
                         [False, False, False, False,  True],
                         [False, False, False, False, False]]])
            and make sure to set the correct dtype and device
        """
        batch_size, seq_len = tokens.size()
        causal_mask = torch.ones((seq_len,seq_len), dtype=torch.bool)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        return causal_mask.unsqueeze(0).to(tokens.device) #NEWLY ADDED

    def get_src_embeddings(self, src: Tensor) -> Tensor:
        """
        Get the non-contextualized source embeddings
        src: torch.Tensor, [batch_size, src_seq_len]
        return: torch.Tensor, [batch_size, src_seq_len, d_model]
        """
        src_x = self.src_embed(src)
        if self.src_pe is not None:
            src_x = self.src_pe(src_x)
        return src_x

    def get_tgt_embeddings(self, tgt: Tensor) -> Tensor:
        """
        Get the non-contextualized target embeddings
        tgt: torch.Tensor, [batch_size, tgt_seq_len]
        return: torch.Tensor, [batch_size, tgt_seq_len, d_model]
        """
        tgt_x = self.tgt_embed(tgt)
        if self.tgt_pe is not None:
            tgt_x = self.tgt_pe(tgt_x)
        return tgt_x

    def forward(self, src: Tensor, tgt: Tensor, normalize_logits: bool = False) -> Tensor:
        """
        Hint: consider 1) the src and tgt shapes, 2) the required masking shapes,
        and, 3) the required attention mechanism shapes.
        This will tell you what the sames are for the intermediate encoder and decoder inputs and outputs are.

        src: torch.Tensor, [batch_size, src_seq_len]
        tgt: torch.Tensor, [batch_size, tgt_seq_len]
        normalize_logits: bool, whether to apply log_softmax to the logits
        return: torch.Tensor, [batch_size, tgt_seq_len, tgt_vocab_size] of logits or log probabilities
        """
        src_embeddings = self.get_src_embeddings(src)
        tgt_embeddings = self.get_tgt_embeddings(tgt)
        #NEED TO COMPLETE
        encoder_mask = self.create_pad_mask(src)
        decoder_pad_mask = self.create_pad_mask(src)
        decoder_causal_mask = self.create_causal_mask(tgt)
        
        encoder_mask = encoder_mask.expand((src.size(0), src.size(1), src.size(1)))
        decoder_causal_mask = decoder_causal_mask.expand(
                (tgt.size(0), -1, -1))
        decoder_pad_mask = decoder_pad_mask.expand(
                (-1, tgt.size(1), -1))

        encoder_values = self.encoder(src_embeddings, encoder_mask)
        return self.decoder(tgt_embeddings, decoder_causal_mask, 
                        encoder_values, decoder_pad_mask, normalize_logits)

    # decoding methods
    @staticmethod
    def all_finished(current_generation: Tensor, eos_token: int, max_len: int = 100) -> bool:
        """
        Check if all the current generation is finished
        current_generation: torch.Tensor, [batch_size, seq_len]
        eos_token: int, the end of sequence token
        max_len: int, the maximum length of the output sequence
        return: bool, True if all the current generation is finished
        """
        return (
            torch.all(torch.any(current_generation == eos_token, dim=-1), dim=0).item()
            or current_generation.shape[-1] >= max_len
        )

    @staticmethod
    def initialize_generation_sequence(src: Tensor, target_sos: int) -> Tensor:
        """
        Initialize the generation by returning the initial input for the decoder.
        src: torch.Tensor, [batch_size, src_seq_len]
        target_sos: int, the start of sequence token
        return: torch.Tensor, [batch_size, 1] on the same device as src and of time int64 filled with target_sos
        """
        return (
            torch.zeros(src.shape[0], 1).fill_(target_sos).type_as(src).to(src.device)
        )

    @staticmethod
    def concatenate_generation_sequence(tgt_generation: Tensor, next_token: Tensor) -> Tensor:
        """
        Concatenate the next token to the current generation
        tgt_generation: torch.Tensor, [batch_size, n, seq_len]
        next_token: torch.Tensor, [batch_size, 1]
        return: torch.Tensor, [batch_size, seq_len + 1]
        """
        return torch.cat([tgt_generation, next_token], dim=-1)

    def pad_generation_sequence(self, tgt_generation: Tensor, target_eos: int) -> Tensor:
        """
        Replace the generation past the end of sequence token with the end of sequence token.

        This is useful for:
            1) finalizing the generation for greedy decoding
            2) helping with intermediate steps in beam search

        tgt_generation: torch.Tensor, [batch_size, seq_len] or [batch_size, k * k, seq_len]
        target_eos: int, the end of sequence token
        return: torch.Tensor, [batch_size, seq_len] or [batch_size, k * k, seq_len]
        """
        lengths = (
            tgt_generation == target_eos
        )  # [batch_size, seq_len] or [batch_size, k * k, seq_len]
        # deal with case where eos was not found  [batch_size, seq_len + 1] or [batch_size, k * k, seq_len + 1]

        lengths = torch.cat(
            [lengths, torch.ones(*lengths.shape[:-1], 1).bool().to(lengths.device)],
            dim=-1,
        )
        # find the first eos token
        a = (
            torch.arange(lengths.shape[-1], device=tgt_generation.device)
            .unsqueeze(0)
            .int()
        )  # [1, seq_len + 1]
        if len(lengths.shape) == 3:
            a = a.unsqueeze(1)
        lengths = (
            torch.where(lengths, lengths * a, torch.ones_like(lengths) * torch.inf)
        ).min(dim=-1)[0]
        # replace tokens past the eos token with the padding token
        mask = a[..., :-1] > lengths.unsqueeze(
            -1
        )  # [batch_size, seq_len] or [batch_size, k * k, seq_len]
        return tgt_generation.masked_fill(mask, self.padding_idx)

    def greedy_decode(self, src: Tensor, target_sos: int, target_eos: int, max_len: int = 100) -> Tensor:
        """
        Do not call the encoder more than once, or you will lose marks.
        The model calls must be batched and the only loop should be over the sequence length, or you will lose marks.
        It will also make evaluating the debugging the model difficult if you do not follow these instructions.

        Hint: use torch.argmax to get the most likely token at each step and
        concatenate_generation_sequence to add it to the sequence.

        src: torch.Tensor, [batch_size, src_seq_len]
        target_sos: int, the start of sequence token
        target_eos: int, the end of sequence token
        max_len: int, the maximum length of the output sequence
        return: torch.Tensor, [batch_size, seq_len]
            Such that each sequence is padded with the padding token after the end of sequence token (if it exists)
            Hint: use the pad_generation_sequence function
        """
        src_embeddings = self.get_src_embeddings(src)
        encoder_mask = self.create_pad_mask(src)
        encoder_mask = encoder_mask.expand((src.size(0), src.size(1), src.size(1)))
        encoder_values = self.encoder(src_embeddings, encoder_mask)

        target_generation = self.initialize_generation_sequence(src, target_sos)

        for i in range(max_len):
            tgt_embeddings = self.get_tgt_embeddings(target_generation)
            decoder_causal_mask = self.create_causal_mask(target_generation)
            decoder_causal_mask = decoder_causal_mask.expand(
                (target_generation.size(0), -1, -1)
            )
            decoder_pad_mask = self.create_pad_mask(src)
            decoder_pad_mask = decoder_pad_mask.expand(
                (-1, target_generation.size(1), -1)
            )
            decoder_values = self.decoder(tgt_embeddings, decoder_causal_mask, 
                        encoder_values, decoder_pad_mask, normalize_logits=True)
            print(decoder_values[:, -1, :])
            next= torch.argmax(decoder_values[:, -1, :], dim=-1, keepdim=True)
            target_generation = self.concatenate_generation_sequence(target_generation, next)
            if self.all_finished(target_generation, target_eos, max_len):
                break
        return self.pad_generation_sequence(target_generation, target_eos)

    @staticmethod
    def expand_encoder_for_beam_search(src_x: Tensor, src_mask: Tensor, k: int) -> Tuple[Tensor, Tensor]:
        """
        Beamsearch will process `batches` of size `batch_size * k` so we need to expand the encoder outputs
        so that we can process the beams in parallel.

        Expand the encoder outputs for beam search to be of size [batch_size * k, ...]
        src_x: torch.Tensor, [batch_size, src_seq_len, d_model]
        src_mask: torch.Tensor, [batch_size, 1, src_seq_len]
        k: int, the beam size
        return: torch.Tensor, [batch_size * k, src_seq_len, d_model], [batch_size * k, 1, src_seq_len]
        """
        batch_size, src_seq_len, d_model = src_x.size()
        expanded_x = src_x.unsqueeze(1).expand(batch_size, k, src_seq_len, d_model)
        expanded_x = expanded_x.reshape(batch_size * k, src_seq_len, d_model)
        expanded_mask = src_mask.unsqueeze(1).expand(batch_size, k, 1, src_seq_len)
        expanded_mask = expanded_mask.reshape(batch_size *k, 1, src_seq_len)
        return (expanded_x, expanded_mask)

    @staticmethod
    def repeat_and_reshape_for_beam_search(t: Tensor, k: int, expan: int, batch_size: int) -> Tensor:
        """
        Repeat the tensor for beam search expan times to be of size [batch_size * k, expan, cur_len]
        and then reshape to [batch_size, k * expan, cur_len]

        t: torch.Tensor, [batch_size * k, cur_len]
        k: int, the beam size
        expan: int, the expansion size
        batch_size: int, the batch size
        return: torch.Tensor, [batch_size, k * expan, cur_len]
        """
        expanded_t = t.unsqueeze(1).expand(batch_size * k, expan, t.size(-1))
        return expanded_t.reshape(batch_size, k * expan, t.size(-1))
        
    def initialize_beams_for_beam_search(
        self, src: Tensor, target_sos: int, target_eos: int, k: int = 5
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        This function will initialize the beam search by taking the first decoder step and using the top-k outputs
        to initialize the beams.

        Here we want to end up with a tensor of shape [batch_size * k, 2] for the input token sequence
        and a tensor of shape [batch_size * k, 2] of log probabilities of the sequences.
        2 is the sequence dimension and is 2 because of the sos token and the first real token.

        This involves the following steps:

            1) Initializes the input sequence with the start of sequence token  with shape [batch_size, 1]
            2) Takes the first step of the decoder and the get the log probabilities, [batch_size, 1, vocab_size]
            3) Please ensure that the end of sequence token is not predicted in the first step.
                Hint: set the log probabilities of the end of sequence token to -inf
            4) Gets the top-k predictions, [batch_size, k]
            5) Initializes the log probabilities of the sequences, [batch_size * k, 1]
            6) Creates the beam tensors with the top-k predictions and log probabilities, [batch_size  * k, 2]
               (i.e., two tensors with this shape).
            7) Expands the encoder outputs for beam search to be of size [batch_size * k, ...]
                Hint: use the expand_encoder_for_beam_search function

        src: torch.Tensor, [batch_size, src_seq_len]
        target_sos: int, the start of sequence token
        target_eos: int, the end of sequence token
        return: torch.Tensor, [batch_size * k, 2], the token sequences
                torch.Tensor, [batch_size * k, 2], the log probabilities of the sequences
                torch.Tensor, [batch_size * k, src_seq_len, d_model], the expanded encoder outputs
                torch.Tensor, [batch_size * k, 1, src_seq_len], the expanded encoder mask
        """
        #Initialize input sequence
        batch_size = src.size(0)
        input_sequence = torch.ones((batch_size, 1), device=src.device, dtype=torch.long)
        input_sequence *= target_sos

        #Take first step of decoder
        src_embeddings = self.get_src_embeddings(src)
        encoder_mask = self.create_pad_mask(src)
        encoder_mask = encoder_mask.expand((batch_size, src.size(1), src.size(1)))
        encoder_values = self.encoder(src_embeddings, encoder_mask)

        tgt_embeddings = self.get_tgt_embeddings(input_sequence)
        decoder_causal_mask = self.create_causal_mask(input_sequence)
        decoder_causal_mask = decoder_causal_mask.expand(
                (input_sequence.size(0), -1, -1)
            )
        decoder_pad_mask = self.create_pad_mask(src)
        decoder_pad_mask = decoder_pad_mask.expand(
                (-1, input_sequence.size(1), -1)
            )
        decoder_values = self.decoder(tgt_embeddings, decoder_causal_mask, 
                        encoder_values, decoder_pad_mask, normalize_logits=True)
        # ARE THE DECODER VALUES THE LOG PROBABILITIES OR DO WE DO THE FOLLOWING?
        # EOS Token check
        decoder_values = decoder_values[:, -1, :]
        decoder_values[:, target_eos] = float('-inf')

        #Top-k predictions
        top_k_values, top_k_indices = torch.topk(decoder_values, k= k, dim=-1)
        #Initialize log probablities
        sequence_log_probabilities = torch.zeros(batch_size*k, 1, device=src.device)
        # sequence_log_probabilities += top_k_values.view(-1, 1)
        sequence_log_probabilities = torch.cat([sequence_log_probabilities, top_k_values.view(-1, 1)], dim=1)

        #Creating beam tensors
        top_k_tensor = torch.cat([input_sequence.repeat(1, k).view(-1, 1), top_k_indices.view(-1, 1)], dim=1) 
        #Expand encoder outputs
        expanded_encoder_outputs, expanded_encoder_mask = self.expand_encoder_for_beam_search(encoder_values, decoder_pad_mask, k)
        return  top_k_tensor, sequence_log_probabilities, expanded_encoder_outputs, expanded_encoder_mask

    def score_sequence_for_beam_search(
        self, tgt_generation: Tensor, seq_log_probs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        This function will get the score of each sequence by summing the log probabilities.

        Note assume that we want this to work for generic shapes of the input where the last dimension is the sequence.

        tgt_generation: torch.Tensor, [..., seq_len]
        seq_log_probs: torch.Tensor, [..., seq_len] 
        return: torch.Tensor, [...., seq_len], the log probabilities of the sequences
                torch.Tensor, [...], the summed scores of the sequences
        """
        score = torch.sum(seq_log_probs, dim=-1)
        return seq_log_probs, score

    def finalize_beams_for_beam_search(self, top_beams: Tensor, device: torch.device) -> Tensor:
        """
        This function will take a list of top beams of length batch_size, where each element is a tensor of some length
        and return a padded tensor of the top beams.  Use self.padding_idx for the padding.

        top_beams: list of torch.Tensor, each of shape [seq_len_i]
        device: torch.device, the device to put the tensor on
        return: torch.Tensor, [batch_size, max_seq_len]
        """
        batch_size = len(top_beams)
        max_seq_len = 0
        for i in top_beams:
            if i.size(0) > max_seq_len:
                max_seq_len = i.size(0)
        padded_top_beams = torch.full((batch_size, max_seq_len), 
                    fill_value=self.padding_idx, device=device, 
                    dtype=top_beams[0].dtype)
        i = 0 
        for beam in top_beams:
            padded_top_beams[i, :beam.size(0)] = beam  
            i += 1
        return padded_top_beams


    def beam_search_decode(self, src: Tensor, target_sos: int, target_eos: int,
                           max_len: int = 100, k: int = 5) -> Tensor:
        """
        This processes the batch in parallel.  Finished beams are removed from the batch and not processed further.

        This needs to keep track of a list of finished candidates along with currently alive beams to prevent
        the search from stopping prematurely with duplicate beams.

        src: torch.Tensor, [batch_size, src_seq_len]
        target_sos: int, the start of sequence token
        target_eos: int, the end of sequence token
        max_len: int, the maximum length of the output sequence
        return: torch.Tensor, [batch_size, max_seq_len] of the mostly likely beam
            Such that each sequence is padded with the padding token after the end of sequence token (if it exists)
        """

        # get the initial first predictions and initialize the beams with them
        batch_size = src.shape[0]  # original batch size
        (
            tgt_generation,
            seq_log_probs,
            src_x,
            src_mask,
        ) = self.initialize_beams_for_beam_search(
            src, target_sos, target_eos, k
        )

        finished_sequences = [[] for _ in range(batch_size)]
        finished_scores = [
            [] for _ in range(batch_size)
        ]  # averaged log probs i.e score
        is_finished = [False] * batch_size

        # use to keep track of which sequence in batch are still being processed
        cur_sentence_ids = torch.arange(batch_size, device=src.device)  # [batch_size]
        cur_batch_size = batch_size

        # expansion size for each beam, 2 so that we can possibly get k finished sequences and k alive
        # this is used in place of vocab_size for efficiency
        expan = 2 * k
        while not all(is_finished) and tgt_generation.shape[-1] < max_len:
            tgt_mask = self.create_causal_mask(tgt_generation)
            tgt_x = self.get_tgt_embeddings(tgt_generation)

            log_probs = self.decoder(
                tgt_x, tgt_mask, src_x, src_mask, normalize_logits=True
            )
            log_probs, pred = torch.topk(
                log_probs[:, -1, :], expan, dim=1
            )  # [batch_size * k, expan]

            # reshape to cur to [batch_size, k * expan, cur_len] and new to [batch_size, k * expan, 1]
            tgt_generation = self.repeat_and_reshape_for_beam_search(
                tgt_generation, k, expan, cur_batch_size
            )
            seq_log_probs = self.repeat_and_reshape_for_beam_search(
                seq_log_probs, k, expan, cur_batch_size
            )
            log_probs = log_probs.reshape(cur_batch_size, k * expan, 1)
            pred = pred.reshape(cur_batch_size, k * expan, 1)

            # expansion to [batch_size, k * expan, cur_len + 1]
            tgt_generation = self.concatenate_generation_sequence(tgt_generation, pred)
            seq_log_probs = torch.cat(
                [seq_log_probs, log_probs], dim=-1
            )  # or concatenate_generation_sequence

            #  score by summing
            seq_log_probs, scores= self.score_sequence_for_beam_search(tgt_generation, seq_log_probs)

            # sort and get top k of the current expan candidates
            scores, topk = torch.topk(scores, expan, dim=1)  # [batch_size, k * expan]

            # gather the top expan candidates sorted by score, [batch_size, expan, cur_len + 1]
            tgt_generation = torch.gather(
                tgt_generation,
                1,
                topk.unsqueeze(-1).expand(-1, -1, tgt_generation.shape[-1]),
            )
            seq_log_probs = torch.gather(
                seq_log_probs,
                1,
                topk.unsqueeze(-1).expand(-1, -1, seq_log_probs.shape[-1]),
            )

            # split into finished and alive
            has_eos = (tgt_generation == target_eos).any(dim=-1)  # [batch_size, expan]
            alive_idxs = []
            has_not_finished_idxs = []  # otherwise remove from batch for efficiency
            for b in range(cur_batch_size):
                original_b = cur_sentence_ids[b].item()  # original sentence id
                alive_idxs_b = []
                for beam in range(expan):
                    if has_eos[b, beam]:
                        finished_sequences[original_b].append(
                            tgt_generation[b, beam, ...].cpu()
                        )
                        finished_scores[original_b].append(scores[b, beam].item())
                    else:
                        alive_idxs_b.append(beam)

                if len(finished_sequences[original_b]) > 1:  # sort finished sequences
                    z = list(
                        zip(finished_scores[original_b], finished_sequences[original_b])
                    )
                    z = sorted(z, key=lambda x: x[0], reverse=True)
                    finished_scores[original_b] = [
                        x[0] for x in z[:k]
                    ]  # cut to k if more than k finished
                    finished_sequences[original_b] = [x[1] for x in z[:k]]

                # these conditions only work assuming an monotonic increase in score (length normalization breaks it)
                has_finished = False
                if len(
                    finished_sequences[original_b]
                ):  # if the most probable finished is more probable than top alive
                    best_finished = max(finished_scores[original_b])
                    if best_finished > scores[b, alive_idxs_b[0]]:
                        has_finished = True
                elif (
                    len(finished_sequences[original_b]) == k
                ):  # if all finished more probable than top alive
                    # This block will never execute, since it is a subset of the if condition assuming that k != 0.
                    # Arvie: This is redundant but should not cause incorrect output. This alternative stopping
                    # condition was left in 1) potentially we could have returned k finished sequences, as is more
                    # traditional for a beamsearch function, 2) I was trying alternative scoring methods which
                    # required this stopping condition.  Apologies if it made the code more confusing.
                    worst_finished = min(finished_scores[original_b])
                    if worst_finished > scores[b, alive_idxs_b[0]]:
                        has_finished = True
                if not has_finished:
                    has_not_finished_idxs.append(b)
                    alive_idxs.append(alive_idxs_b[:k])

            if len(has_not_finished_idxs) == 0:  # all finished
                break

            # gather sequences that still need decoding, this is done for efficiency
            has_not_finished_idxs = torch.tensor(
                has_not_finished_idxs, dtype=torch.long, device=src.device
            )
            tgt_generation = torch.index_select(
                tgt_generation, 0, has_not_finished_idxs
            )
            seq_log_probs = torch.index_select(seq_log_probs, 0, has_not_finished_idxs)
            src_shape = src_x.shape
            src_x = src_x.reshape(cur_batch_size, k, src_shape[-2], src_shape[-1])
            src_mask = src_mask.reshape(cur_batch_size, k, -1)
            src_x = torch.index_select(src_x, 0, has_not_finished_idxs)
            src_mask = torch.index_select(src_mask, 0, has_not_finished_idxs)
            cur_batch_size = len(has_not_finished_idxs)
            src_x = src_x.reshape(cur_batch_size * k, src_shape[-2], src_shape[-1])
            src_mask = src_mask.reshape(cur_batch_size * k, 1, src_shape[-2])
            cur_sentence_ids = torch.index_select(
                cur_sentence_ids, 0, has_not_finished_idxs
            )

            # gather the top k alive beams
            alive_idxs = torch.tensor(
                alive_idxs, dtype=torch.long, device=tgt_generation.device
            )

            tgt_generation = torch.gather(
                tgt_generation,
                1,
                alive_idxs.unsqueeze(-1).expand(-1, -1, tgt_generation.shape[-1]),
            )
            seq_log_probs = torch.gather(
                seq_log_probs,
                1,
                alive_idxs.unsqueeze(-1).expand(-1, -1, seq_log_probs.shape[-1]),
            )

            # reshape to [batch_size * k, cur_len]
            tgt_generation = tgt_generation.reshape(cur_batch_size * k, -1)
            seq_log_probs = seq_log_probs.reshape(cur_batch_size * k, -1)

        if not all(is_finished):  # take top alive if no finished
            tgt_generation = tgt_generation.reshape(cur_batch_size, k, -1)
            for b in range(cur_batch_size):
                original_b = cur_sentence_ids[b].item()
                if len(finished_sequences[original_b]) == 0:  # just take top alive
                    g = tgt_generation[b, 0, ...]
                    finished_sequences[original_b].append(g.cpu())

        return self.finalize_beams_for_beam_search(
            [x[0] for x in finished_sequences], src_x.device
        )

    def beam_search_decode_slow(self, src: Tensor, target_sos: int, target_eos: int,
                                max_len: int = 100, k: int = 5) -> Tensor:
        """
        Slow version which does not batch the beam search. This is useful for confirming debugging and understanding.

        Consider the summed log probability of the sequence (including eos) when scoring and sorting the candidates.

        This needs to keep track of a list of finished candidates along with currently alive beams to prevent
        the search from stopping prematurely with duplicate beams.

        src: torch.Tensor, [batch_size, src_seq_len]
        target_sos: int, the start of sequence token
        target_eos: int, the end of sequence token
        max_len: int, the maximum length of the output sequence
        return: torch.Tensor, [batch_size, max_seq_len] of the mostly likely beam
            Such that each sequence is padded with the padding token after the end of sequence token (if it exists)
        """

        # print('Warning: this is a slow version of beam search and should only be run for debugging.')
        # get the initial first predictions and initialize the beams with them
        batch_size = src.shape[0]  # original batch size

        (
            tgt_generation,
            seq_log_probs,
            src_x,
            src_mask,
        ) = self.initialize_beams_for_beam_search(
            src, target_sos, target_eos, k
        )

        # reshape so we can loop over the batch dimension
        tgt_generation = tgt_generation.reshape(batch_size, k, -1)
        seq_log_probs = seq_log_probs.reshape(batch_size, k, -1)
        src_x = src_x.reshape(batch_size, k, src_x.shape[-2], src_x.shape[-1])
        src_mask = src_mask.reshape(
            batch_size, k, 1, -1
        )  # note the extra dim for the mask

        top_beams = []
        expan = 2 * k  # expansion size, 2 * k for k finished and k alive,
        # this is used in place of vocab_size for efficiency
        for b in range(batch_size):
            # slice out the current sequence from the batch
            tgt_generation_b = tgt_generation[b, ...]
            seq_log_probs_b = seq_log_probs[b, ...]
            src_x_b = src_x[b, ...]
            src_mask_b = src_mask[b, ...]

            finished_sequences_b = []
            finished_scores_b = []

            for i in range(1, max_len):
                tgt_mask = self.create_causal_mask(tgt_generation_b)
                tgt_x = self.get_tgt_embeddings(tgt_generation_b)
                log_probs = self.decoder(
                    tgt_x, tgt_mask, src_x_b, src_mask_b, normalize_logits=True
                )
                log_probs, pred = torch.topk(log_probs[:, -1, :], expan, dim=1)

                # reshape to cur to [k * expan, cur_len] and new to [k * expan, 1]
                tgt_generation_b = self.repeat_and_reshape_for_beam_search(
                    tgt_generation_b, k, expan, 1
                ).squeeze(0)
                seq_log_probs_b = self.repeat_and_reshape_for_beam_search(
                    seq_log_probs_b, k, expan, 1
                ).squeeze(0)
                log_probs = log_probs.reshape(k * expan, 1)
                pred = pred.reshape(k * expan, 1)

                # expansion to [batch_size, k * expan, cur_len + 1]
                tgt_generation_b = self.concatenate_generation_sequence(
                    tgt_generation_b, pred
                )
                seq_log_probs_b = torch.cat([seq_log_probs_b, log_probs], dim=-1)

                # masking to length and score by summing
                tgt_generation_b = self.pad_generation_sequence(tgt_generation_b, target_eos)
                seq_log_probs_b, scores = self.score_sequence_for_beam_search(tgt_generation_b, seq_log_probs_b)

                # sort and get top k of the current expan candidates
                scores, topk = torch.topk(scores, expan, dim=0)  # [k * expan]

                # gather the top expan candidates sorted by score, [expan, cur_len + 1]
                tgt_generation_b = torch.index_select(tgt_generation_b, 0, topk)
                seq_log_probs_b = torch.index_select(seq_log_probs_b, 0, topk)

                # split into finished and alive and determine if we are finished
                has_eos = (tgt_generation_b == target_eos).any(dim=-1)  # [expan]
                alive_idxs_b = []
                for beam in range(expan):
                    if has_eos[beam]:
                        finished_sequences_b.append(tgt_generation_b[beam, ...].cpu())
                        finished_scores_b.append(scores[beam].item())
                    else:
                        alive_idxs_b.append(beam)
                if len(finished_sequences_b) > 1:  # sort finished sequences
                    z = list(zip(finished_scores_b, finished_sequences_b))
                    z = sorted(z, key=lambda x: x[0], reverse=True)
                    finished_scores_b = [
                        x[0] for x in z[:k]
                    ]  # cut to k if more than k finished
                    finished_sequences_b = [x[1] for x in z[:k]]

                # these conditions only work assuming an monotonic increase in score (length normalization breaks it)
                if len(
                    finished_sequences_b
                ):  # finished if most probable finished is more probable than top alive
                    best_finished = max(finished_scores_b)
                    if best_finished > scores[alive_idxs_b[0]]:
                        break
                elif (
                    len(finished_sequences_b) == k
                ):  # finished if all finished more probable than top alive
                    worst_finished = min(finished_scores_b)
                    if worst_finished > scores[alive_idxs_b[0]]:
                        break

                # gather the top k alive beams
                alive_idxs_b = torch.tensor(
                    alive_idxs_b[:k], dtype=torch.long, device=tgt_generation.device
                )
                tgt_generation_b = torch.index_select(tgt_generation_b, 0, alive_idxs_b)
                seq_log_probs_b = torch.index_select(seq_log_probs_b, 0, alive_idxs_b)

            if len(finished_sequences_b) == 0:  # take top alive if no finished
                finished_sequences_b.append(tgt_generation_b[0, ...].cpu())
            top_beams.append(finished_sequences_b[0])

        return self.finalize_beams_for_beam_search(top_beams, src_x.device)

    def store_attention_scores(self, should_store: bool = True) -> None:
        self.decoder.store_attention_scores(should_store)

    def get_attention_scores(self):
        return self.decoder.get_attention_scores()
