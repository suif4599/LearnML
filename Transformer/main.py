import os
import torch
from core import *
# from test_dataset import gen_data
print(f"cuda is {'available' if torch.cuda.is_available() else 'not available'}")

from data import gen_dict, tokenize

eng_vocab, chn_vocab, eng_len, chn_len, train_loader, test_loader = \
    gen_dict(SOS, EOS, PAD, UNK, START_INDEX, 
             batch_size=32, min_freq=2,
             max_eng_len=10, max_chn_len=15)


# filename = os.path.join(os.path.dirname(__file__), "model_save", "model_20250304200157_epoch4.pt")
# model = TransformerTranslator.load(filename, eng_vocab, chn_vocab, eng_len, chn_len, 1024, 16, 8, dropout=0.2)
model = TransformerTranslator(eng_vocab, chn_vocab, eng_len, chn_len, 512, 8, 8, dropout=0.2)
sample = ["I am a student.",
          "I'd like to have a cup of coffee.",
          "I want to go to the library.",
          "Can you help me with my homework?",
          "I am not feeling well today.",
          "Tom gave me a book as a birthday present.",
          "I am looking forward to seeing you.",
          "I am sorry for being late.",
          "I am happy to see you again.",
          "I am so excited about the concert."]

model.start_training(train_loader, test_loader, 16, torch.optim.Adam(model.parameters(), lr=0.0001),
                     save_path=os.path.join(os.path.dirname(__file__), "model_save"), only_save_params=True)
for s in sample:
    print("English:", s)
    print("Chinese:", model.translate(s))
    print()
while 1:
    model.eval()
    eng = input("Input English sentence: ").strip()
    if eng == "exit":
        break
    print("Chinese translation:", model.translate(eng))
    print()

