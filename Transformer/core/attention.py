import torch
import torch.nn.functional as F

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, n_head, seq_len=-1, future_mask=False):
        super().__init__()
        self.__d_model = d_model
        self.__n_head = n_head
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        self.__d_k = d_model // n_head
        self.__sqrt_d_k = self.__d_k ** 0.5
        self.__d_v = d_model // n_head
        self.WQ = torch.nn.Linear(d_model, d_model, bias=False)
        self.WK = torch.nn.Linear(d_model, d_model, bias=False)
        self.WV = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_Qi = torch.nn.ModuleList([torch.nn.Linear(d_model, self.__d_k, bias=False) for _ in range(n_head)])
        self.W_Ki = torch.nn.ModuleList([torch.nn.Linear(d_model, self.__d_k, bias=False) for _ in range(n_head)])
        self.W_Vi = torch.nn.ModuleList([torch.nn.Linear(d_model, self.__d_v, bias=False) for _ in range(n_head)])
        self.output_linear = torch.nn.Linear(n_head * self.__d_v, d_model, bias=False)
        self.__seq_len = seq_len
        if future_mask:
            future_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8), diagonal=1).\
                unsqueeze(0).unsqueeze(0).expand(1, n_head, -1, -1)
            self.register_buffer("future_mask", future_mask.clone())
    
    def forward(self, x, mask=None):
        """
        x: (batch_size, seq_len, d_model)
        mask: (batch_size, seq_len)
        """
        if self.__seq_len > 0:
            if mask is not None:
                mask = mask.unsqueeze(1).unsqueeze(-1).expand(-1, self.__n_head, -1, self.__seq_len)
                mask = mask + self.future_mask
            else:
                mask = self.future_mask
        else:
            mask = mask.unsqueeze(1).unsqueeze(-1)
        q = self.WQ(x).view(x.size(0), x.size(1), self.__n_head, self.__d_k).transpose(1, 2)
        # q: (batch_size, n_head, seq_len, d_k)
        k_T = self.WK(x).view(x.size(0), x.size(1), self.__n_head, self.__d_k).permute(0, 2, 3, 1)
        # k_T: (batch_size, n_head, d_k, seq_len)
        v = self.WV(x).view(x.size(0), x.size(1), self.__n_head, self.__d_v).transpose(1, 2)
        # v: (batch_size, n_head, seq_len, d_v)
        scores = q @ k_T / self.__sqrt_d_k
        # scores: (batch_size, n_head, seq_len, seq_len)
        if mask is not None:
            scores = scores.masked_fill((mask == 1), -1e9)
        attention = F.softmax(scores, dim=-1)
        context = attention @ v
        context = context.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.__d_model)
        return self.output_linear(context)
    
class EncoderDecoderAttention(torch.nn.Module):
    def __init__(self, d_model, n_head, seq_len):
        super().__init__()
        self.__d_model = d_model
        self.__n_head = n_head
        self.__seq_len = seq_len
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        self.__d_k = d_model // n_head
        self.__sqrt_d_k = self.__d_k ** 0.5
        self.__d_v = d_model // n_head
        future_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8), diagonal=1).\
            unsqueeze(0).unsqueeze(0).expand(1, n_head, -1, -1)
        self.register_buffer("future_mask", future_mask.clone())
        self.WQ = torch.nn.Linear(d_model, d_model, bias=False)
        self.WK = torch.nn.Linear(d_model, d_model, bias=False)
        self.WV = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_Qi = [torch.nn.Linear(d_model, self.__d_k, bias=False) for _ in range(n_head)]
        self.W_Ki = [torch.nn.Linear(d_model, self.__d_k, bias=False) for _ in range(n_head)]
        self.W_Vi = [torch.nn.Linear(d_model, self.__d_v, bias=False) for _ in range(n_head)]
        self.output_linear = torch.nn.Linear(n_head * self.__d_v, d_model, bias=False)
    
    def forward(self, x, encoder_output, mask=None):
        """
        x: (batch_size, seq_len, d_model)
        encoder_output: (batch_size, seq_len, d_model)
        mask: (batch_size, seq_len)
        """
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(-1).expand(-1, self.__n_head, -1, self.__seq_len)
            mask = mask + self.future_mask
        else:
            mask = self.future_mask
        q = self.WQ(x).view(x.size(0), x.size(1), self.__n_head, self.__d_k).transpose(1, 2)
        # q: (batch_size, n_head, seq_len, d_k)
        k_T = self.WK(encoder_output).view(encoder_output.size(0), encoder_output.size(1), self.__n_head, self.__d_k).permute(0, 2, 3, 1)
        # k_T: (batch_size, n_head, d_k, seq_len)
        v = self.WV(encoder_output).view(encoder_output.size(0), encoder_output.size(1), self.__n_head, self.__d_v).transpose(1, 2)
        # v: (batch_size, n_head, seq_len, d_v)
        scores = q @ k_T / self.__sqrt_d_k
        # scores: (batch_size, n_head, seq_len, seq_len)
        if mask is not None:
            mask = mask[:, :, :, :scores.size(-1)]
            scores = scores.masked_fill(mask != 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        context = attention @ v
        context = context.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.__d_model)
        return self.output_linear(context)