import torch
from .attention import MultiHeadAttention

class PositionalEncodingLayer(torch.nn.Module):
    def __init__(self, d_model, seq_len):
        super().__init__()
        self.__d_model = d_model
        self.__seq_len = seq_len
        self.matrix = torch.zeros((seq_len, d_model))
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                self.matrix[pos, i] = torch.sin(pos / (10000 ** (i / d_model)))
                self.matrix[pos, i + 1] = torch.cos(pos / (10000 ** (i / d_model)))
        self.matrix = self.matrix.unsqueeze(0)

    def forward(self, x):
        return x + self.matrix

class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.__d_model = d_model
        self.__n_head = n_head
        self.W_Q = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_K = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_V = torch.nn.Linear(d_model, d_model, bias=False)
        self.multi_head_attention = MultiHeadAttention(d_model, n_head)
        self.add_and_norm1 = torch.nn.LayerNorm(d_model)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(d_model, 4 * d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * d_model, d_model)
        )
        self.add_and_norm2 = torch.nn.LayerNorm(d_model)
    
    def forward(self, x):
        x = self.add_and_norm1(x + self.multi_head_attention(self.W_Q(x), self.W_K(x), self.W_V(x)))
        x = self.add_and_norm2(x + self.feed_forward(x))
        return x
    
