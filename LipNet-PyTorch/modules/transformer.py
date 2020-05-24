import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import numpy as np


class TransformerModel(nn.Module):

    def __init__(self, input_size, emb_size, nhead, nhid, nlayers):
        """
        emb_size: Embedding Size for the input
        ntoken: Number of tokens Vocab Size
        nhead: Number of transformer heads in the encoder
        nhid: Number of hidden units in transformer encoder layer
        nlayer: Number of layers in transformer encoder
        """
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer

        # Initialized position input embedding, position encoding layers
        # self.encoder = nn.Embedding(ntoken, emb_size)
        self.linear = nn.Linear(input_size, emb_size)
        self.pos_encoding = PositionalEncoding(emb_size)

        # Initialized transformer encoder with nlayers and each layer having nhead heads and nhid hidden units.
        encoder_layers = TransformerEncoderLayer(emb_size, nhead, nhid, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)


    def forward(self, src):
        """
        src: tensor of shape (seq_len, batch_size)

        Returns:
            output: tensor of shape (seq_len, batch_size, vocab_size)
        """
        # Embed the source sequences and add the positional encoding.
        src = self.linear(src)
        src = self.pos_encoding(src)
        output = self.transformer_encoder(src)

        return output


class PositionalEncoding(nn.Module):
    """
    Adds positional embedding to the input for conditioning on time.
    From the paper "Attention is all you need"
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: tensor of shape (seq_len, batch_size, embedding_size)
        Returns:
            x: tensor of shape (seq_len, batch_size, embedding_size)
        """
        x = x + self.pe[:x.size(0), :]
        return x