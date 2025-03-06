import os
import torch
from core import *
# from test_dataset import gen_data
print(f"cuda is {'available' if torch.cuda.is_available() else 'not available'}")

from data import TranslateDataset, Language


# dataset = TranslateDataset(SOS, EOS, PAD, UNK, START_INDEX, 
#                            batch_size=32, min_freq=2, 
#                            max_eng_len=10, max_chn_len=15)
dataset = TranslateDataset(max_rows=100_000, min_eng_freq=5e-5,
                           max_eng_len=32, max_chn_len=64,
                           batch_size=64)


filename = os.path.join(os.path.dirname(__file__), "model_save", "model_20250306080940_epoch4.pt")
# model = TransformerTranslator.load(filename, dataset, 512, 16, 6, dropout=0.1)

model = TransformerTranslator(dataset, 512, 16, 6, dropout=0.15)
model.start_training(30, torch.optim.Adam(model.parameters(), lr=0.0001),
                     save_path=os.path.join(os.path.dirname(__file__), "model_save"), only_save_params=True)

while 1:
    model.eval()
    eng = input("Input English sentence: ").strip()
    if eng == "exit":
        break
    print("Chinese translation:", model.translate(eng))
    print()

