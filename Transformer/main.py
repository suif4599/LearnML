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




# filename = os.path.join(os.path.dirname(__file__), "model_save", "model_20250302222611_epoch10.pth")

# model = TransformerTranslator.load(filename)
# model.eval()

# if torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")
# model.to(device)
# eng_len = model._TransformerTranslator__source_seq_len + 1
# chn_len = model._TransformerTranslator__target_seq_len + 1
# while True:
#     eng = input("Input English sentence: ")
#     if eng == "exit":
#         break
#     eng = tokenize(eng, UNK, eng_vocab, eng_len)
#     eng = eng.to(device).unsqueeze(0)
#     chn = [SOS] + [PAD] * (chn_len - 1)
#     for _ in range(20):
#         chn = torch.tensor([chn]).long().unsqueeze(0).to(device)
#         output = model(eng[:, :-1], chn)
#         output = output.argmax(dim=-1).item()
#         if output == EOS:
#             break
#         chn.append(output)
#     print("Chinese translation:", " ".join([chn_vocab[i] for i in chn[1:]]))

filename = os.path.join(os.path.dirname(__file__), "model_save", "model_20250302225706_epoch14.pt")

model = TransformerTranslator.load(filename, len(eng_vocab), len(chn_vocab), eng_len, chn_len, 512, 8, 6)
model.eval()
chn_rev_map = {v: k for k, v in chn_vocab.items()}
while 1:
    eng = input("Input English sentence: ")
    if eng == "exit":
        break
    chn = ''.join(chn_rev_map[i] for i in model.translate(tokenize(eng, UNK, eng_vocab, eng_len)))
    print("Chinese translation:", chn)
    
