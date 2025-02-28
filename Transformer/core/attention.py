import torch
import torch.nn.functional as F

class DotProductAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, q, k, v, mask=None):
        """
        q, k: (batch_size, n_head, seq_len, d_k)
        v: (batch_size, n_head, seq_len, d_v)
        mask: (batch_size, 1, seq_len, seq_len)
        """
        attention = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        return torch.matmul(attention, v)

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.__d_model = d_model
        self.__n_head = n_head
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        self.__d_k = d_model // n_head
        self.__d_v = d_model // n_head
        self.W_Q = [torch.nn.Linear(d_model, self.__d_k, bias=False) for _ in range(n_head)]
        self.W_K = [torch.nn.Linear(d_model, self.__d_k, bias=False) for _ in range(n_head)]
        self.W_V = [torch.nn.Linear(d_model, self.__d_v, bias=False) for _ in range(n_head)]
        self.attentions = [DotProductAttention() for _ in range(n_head)]
        self.output_linear = torch.nn.Linear(n_head * self.__d_v, d_model, bias=False)
        self.attention = DotProductAttention()
    
    def forward(self, q, k, v, mask=None):
        """
        q, k, v: (batch_size, seq_len, d_model)
        mask: (batch_size, 1, seq_len, seq_len)
        """
        attentions = [attention(q @ W_Q, k @ W_K, v @ W_V, mask) 
                      for attention, W_Q, W_K, W_V in zip(self.attentions, self.W_Q, self.W_K, self.W_V)]
        attentions = torch.cat(attentions, dim=-1)
        return self.output_linear(attentions)