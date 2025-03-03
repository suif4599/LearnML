import os
import torch
from core import *
# from test_dataset import gen_data
print(f"cuda is {'available' if torch.cuda.is_available() else 'not available'}")

from data import gen_dict, tokenize

eng_vocab, chn_vocab, eng_len, chn_len, train_loader, test_loader = \
    gen_dict(SOS, EOS, PAD, UNK, START_INDEX, batch_size=32, min_freq=2)

# model = TransformerTranslator(len(eng_vocab), len(chn_vocab), eng_len, chn_len, 512, 8, 6)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# model.start_training(train_loader, test_loader, 15, optimizer,
#                      save_path=os.path.join(os.path.dirname(__file__), "model_save"), only_save_params=True)


filename = os.path.join(os.path.dirname(__file__), "model_save", "model_20250303163648_epoch3.pt")

model = TransformerTranslator.load(filename, len(eng_vocab), len(chn_vocab), eng_len, chn_len, 512, 8, 6)
# model = TransformerTranslator(len(eng_vocab), len(chn_vocab), eng_len, chn_len, 512, 8, 6)
chn_rev_map = {v: k for k, v in chn_vocab.items()}
model.start_training(train_loader, test_loader, 7, torch.optim.Adam(model.parameters(), lr=0.0001),
                     save_path=os.path.join(os.path.dirname(__file__), "model_save"), only_save_params=True)
while 1:
    model.eval()
    eng = input("Input English sentence: ")
    if eng == "exit":
        break
    chn = ''.join(chn_rev_map[i] for i in model.translate(tokenize(eng, UNK, eng_vocab, eng_len)))
    print("Chinese translation:", chn)

