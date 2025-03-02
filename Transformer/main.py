from test_dataset import gen_data

seq_len = 64
batch_size = 16

train_loader, test_loader, vocab_size = gen_data(batch_size, seq_len)

import torch
from core import TransformerTranslator
print(f"cuda is {'available' if torch.cuda.is_available() else 'not available'}")

model = TransformerTranslator(vocab_size, seq_len, 128, 4, 4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.start_training(train_loader, test_loader, 10, optimizer)


