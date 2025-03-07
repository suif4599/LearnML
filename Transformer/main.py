import os
import torch
from core import *
# from test_dataset import gen_data
print(f"cuda is {'available' if torch.cuda.is_available() else 'not available'}")

from data import TranslateDataset, Language





# dataset = TranslateDataset(SOS, EOS, PAD, UNK, START_INDEX, 
#                            batch_size=32, min_freq=2, 
#                            max_eng_len=10, max_chn_len=15)
                           

filename = os.path.join(os.path.dirname(__file__), "model_save", "TransformerTranslator_20250307145944")

warmup_epoch = 8
lr_max = 1e-4
lr_min = 1e-5

# model = TransformerTranslator.load(filename)
# model.dataset.reload()

dataset = TranslateDataset(max_rows=800_000, min_eng_freq=5e-5,
                           max_eng_len=15, max_chn_len=20,
                           batch_size=64, max_eng_vocab=10_000)
model = TransformerTranslator(dataset, 256, 8, 32, dropout=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr_min)
lambda_lr = torch.optim.lr_scheduler.LambdaLR(optimizer, 
                                              lambda epoch: lr_max / lr_min if epoch >= warmup_epoch else (lr_max / lr_min) ** (epoch / warmup_epoch))
model.start_training(60, optimizer, lambda_lr,
                     save_path=os.path.join(os.path.dirname(__file__), "model_save"))

while 1:
    model.eval()
    eng = input("Input English sentence: ").strip()
    if eng == "exit":
        break
    print("Chinese translation:", model.translate(eng))
    print()

