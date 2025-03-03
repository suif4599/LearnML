import torch
import math
from .attention import MultiHeadAttention, EncoderDecoderAttention

class PositionalEncodingLayer(torch.nn.Module):
    def __init__(self, d_model, seq_len):
        super().__init__()
        self.__d_model = d_model
        self.__seq_len = seq_len
        matrix = torch.zeros((seq_len, d_model))
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                matrix[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
                matrix[pos, i + 1] = math.cos(pos / (10000 ** (i / d_model)))
        matrix = matrix.unsqueeze(0)
        self.register_buffer("matrix", matrix.clone())

    def forward(self, x):
        return x + self.matrix
        

class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.__d_model = d_model
        self.__n_head = n_head
        self.multi_head_attention = MultiHeadAttention(d_model, n_head)
        self.add_and_norm1 = torch.nn.LayerNorm(d_model)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(d_model, 4 * d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * d_model, d_model)
        )
        self.add_and_norm2 = torch.nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        x = self.add_and_norm1(x + self.multi_head_attention(x, mask))
        x = self.add_and_norm2(x + self.feed_forward(x))
        return x
    
class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, n_head, seq_len):
        super().__init__()
        self.__d_model = d_model
        self.__n_head = n_head
        self.__seq_len = seq_len
        self.masked_multi_head_attention = MultiHeadAttention(d_model, n_head, seq_len, True)
        self.add_and_norm1 = torch.nn.LayerNorm(d_model)
        self.encoder_decoder_attention = EncoderDecoderAttention(d_model, n_head, seq_len)
        self.add_and_norm2 = torch.nn.LayerNorm(d_model)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(d_model, 4 * d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * d_model, d_model)
        )
        self.add_and_norm3 = torch.nn.LayerNorm(d_model)
    
    def forward(self, x, encoder_output, mask=None):
        x = self.add_and_norm1(x + self.masked_multi_head_attention(x, mask))
        x = self.add_and_norm2(x + self.encoder_decoder_attention(x, encoder_output, mask))
        x = self.add_and_norm3(x + self.feed_forward(x))
        return x