import torch
from .layer import PositionalEncodingLayer, EncoderLayer

class Encoder(torch.nn.Module):
    "Input: (batch_size, seq_len, vocab_size)"
    def __init__(self, vocab_size, seq_len, d_model, n_head, n_layer):
        super().__init__()
        self.__vocab_size = vocab_size
        self.__seq_len = seq_len
        self.__d_model = d_model
        self.__n_head = n_head
        self.__n_layer = n_layer
        # (batch_size, seq_len, vocab_size)
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        # (batch_size, seq_len, d_model)
        self.pos_encoder = PositionalEncodingLayer(d_model, seq_len)
        # (batch_size, seq_len, d_model)
        self.encoder_layers = [EncoderLayer(d_model, n_head) for _ in range(n_layer)]
        # (batch_size, seq_len, d_model)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        return x