import torch
import sys
torch.manual_seed(0)
sys.path.append("Transformer")
from Transformer.core.layer import EncoderLayer, DecoderLayer

t1 = torch.randint(0, 9, (1, 4))
t2 = torch.randint(0, 9, (1, 4))
mask = torch.Tensor([[0]*2+[1]*2]) == 1
t2[0, 0] = t1[0, 0]
print(t1)
print(t2)
print(t1 == t2)

l1 = torch.nn.Embedding(10, 4)
t1 = l1(t1)
t2 = l1(t2)
print(t1)
print(t2)
print(t1 == t2)

l2 = EncoderLayer(4, 2, dropout=0)
t1 = l2(t1, mask)
t2 = l2(t2, mask)
print(t1)
print(t2)
print(t1 == t2)
