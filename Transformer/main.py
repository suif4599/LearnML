from test_dataset import gen_data

seq_len = 64
batch_size = 16

train_loader, test_loader, vocab_size = gen_data(batch_size, seq_len)

import torch
from core import TransformerTranslator

model = TransformerTranslator(vocab_size, seq_len, 128, 4, 4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.start_training(train_loader, test_loader, 10, optimizer)


