import os
import torch
import warnings
from math import inf
from enum import Enum

def check_existence():
    raw_data = os.path.join(os.path.dirname(__file__), "raw_data")
    if not os.path.exists(raw_data):
        raise FileNotFoundError(f'folder "{raw_data}" found')
    if not os.path.isdir(raw_data):
        raise NotADirectoryError(f'"{raw_data}" is not a directory')
    train_path = os.path.join(raw_data, "train.txt")
    test_path = os.path.join(raw_data, "test.txt")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f'file "{train_path}" not found')
    if not os.path.exists(test_path):
        raise FileNotFoundError(f'file "{test_path}" not found')
    if not os.path.isfile(train_path):
        raise IsADirectoryError(f'"{train_path}" is a directory')
    if not os.path.isfile(test_path):
        raise IsADirectoryError(f'"{test_path}" is a directory')

def gen_dict(SOS, EOS, PAD, UNK, START_INDEX, batch_size=32, _eng_len=-1, _chn_len=-1, 
             *, min_freq=None, max_vocab=None, device=None, 
             max_eng_len=inf, max_chn_len=inf):
    # Check existence
    check_existence()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Read
    import opencc
    cc = opencc.OpenCC('t2s')
    raw_data = os.path.join(os.path.dirname(__file__), "raw_data")
    train_path = os.path.join(raw_data, "train.txt")
    test_path = os.path.join(raw_data, "test.txt")
    with open(train_path, "r") as f:
        train_data = f.readlines()
    train_data = [cc.convert(line.strip()).split('\t') for line in train_data]
    # Tokenize
    train_tokenized = [(eng.lower()[:-1].split() + list(eng[-1]), list(chn)) 
                       for eng, chn in train_data
                       if len(eng.lower().split()) <= max_eng_len and len(chn) <= max_chn_len]
    eng_vocab_cnt = {}
    chn_vocab_cnt = {}
    for eng, chn in train_tokenized:
        for word in eng:
            if word not in eng_vocab_cnt:
                eng_vocab_cnt[word] = 0
            eng_vocab_cnt[word] += 1
        for word in chn:
            if word not in chn_vocab_cnt:
                chn_vocab_cnt[word] = 0
            chn_vocab_cnt[word] += 1
    print(f"English raw vocab size: {len(eng_vocab_cnt)}")
    print(f"Chinese raw vocab size: {len(chn_vocab_cnt)}")
    # Filter
    if min_freq is not None:
        eng_vocab_cnt = {k: v for k, v in eng_vocab_cnt.items() if v >= min_freq}
        chn_vocab_cnt = {k: v for k, v in chn_vocab_cnt.items() if v >= min_freq}
    if max_vocab is not None:
        eng_vocab_cnt = dict(sorted(eng_vocab_cnt.items(), key=lambda x: x[1], reverse=True)[:max_vocab])
        chn_vocab_cnt = dict(sorted(chn_vocab_cnt.items(), key=lambda x: x[1], reverse=True)[:max_vocab])
    print(f"English filtered vocab size: {len(eng_vocab_cnt)}")
    print(f"Chinese filtered vocab size: {len(chn_vocab_cnt)}")
    # Build vocab
    eng_vocab = {word: i for i, word in enumerate(eng_vocab_cnt, START_INDEX)}
    chn_vocab = {word: i for i, word in enumerate(chn_vocab_cnt, START_INDEX)}
    eng_vocab["<sos>"] = chn_vocab["<sos>"] = SOS
    eng_vocab["<eos>"] = chn_vocab["<eos>"] = EOS
    eng_vocab["<pad>"] = chn_vocab["<pad>"] = PAD
    eng_vocab["<unk>"] = chn_vocab["<unk>"] = UNK
    # Build data
    eng_len = max(max(len(eng) for eng, _ in train_tokenized) + 3, _eng_len)
    chn_len = max(max(len(chn) for _, chn in train_tokenized) + 3, _chn_len)
    if eng_len > _eng_len > 0 or chn_len > _chn_len > 0:
        warnings.warn(f"Data length is too short, expand to eng_len={eng_len}, chn_len={chn_len}")
    train_data_eng = []
    train_data_chn = []
    for eng, chn in train_tokenized:
        _eng = [eng_vocab[word] if word in eng_vocab else UNK for word in eng]
        _chn = [chn_vocab[word] if word in chn_vocab else UNK for word in chn]
        _eng = [eng_vocab["<sos>"]] + _eng + [eng_vocab["<eos>"]] + ([eng_vocab["<pad>"]] * (eng_len - len(_eng) - 2))
        _chn = [chn_vocab["<sos>"]] + _chn + [chn_vocab["<eos>"]] + ([chn_vocab["<pad>"]] * (chn_len - len(_chn) - 2))
        train_data_eng.append(_eng)
        train_data_chn.append(_chn)
    train_data = torch.utils.data.TensorDataset(
        torch.Tensor(train_data_eng).long().to(device),
        torch.Tensor(train_data_chn).long().to(device)
    )
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    with open(test_path, "r") as f:
        test_data = f.readlines()
    test_data = [cc.convert(line.strip()).split('\t') for line in test_data]
    test_tokenized = [(eng.lower()[:-1].split() + list(eng[-1]), list(chn)) for eng, chn in test_data]
    test_data_eng = []
    test_data_chn = []
    for eng, chn in test_tokenized:
        _eng = [eng_vocab[word] if word in eng_vocab else UNK for word in eng]
        _chn = [chn_vocab[word] if word in chn_vocab else UNK for word in chn]
        if eng_len - len(_eng) - 2 < 0 or chn_len - len(_chn) - 2 < 0:
            continue
        _eng = [eng_vocab["<sos>"]] + _eng + [eng_vocab["<eos>"]] + ([eng_vocab["<pad>"]] * (eng_len - len(_eng) - 2))
        _chn = [chn_vocab["<sos>"]] + _chn + [chn_vocab["<eos>"]] + ([chn_vocab["<pad>"]] * (chn_len - len(_chn) - 2))
        test_data_eng.append(_eng)
        test_data_chn.append(_chn)
    test_data = torch.utils.data.TensorDataset(
        torch.Tensor(test_data_eng).long().to(device),
        torch.Tensor(test_data_chn).long().to(device)
    )
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=False)
    return eng_vocab, chn_vocab, eng_len, chn_len, train_loader, test_loader


class Language(Enum):
    ENGLISH = 0
    CHINESE = 1

class TranslateDataset(torch.utils.data.Dataset):
    LANGUAGE = Language
    def __init__(self, SOS=0, EOS=1, PAD=2, UNK=3, START_INDEX=4, batch_size=32, eng_len=-1, chn_len=-1, 
                 *, min_freq=None, max_vocab=None, device=None, 
                 max_eng_len=inf, max_chn_len=inf):
        self.SOS = SOS
        self.EOS = EOS
        self.PAD = PAD
        self.UNK = UNK
        self.START_INDEX = START_INDEX
        self.batch_size = batch_size
        self.min_freq = min_freq
        self.max_vocab = max_vocab
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_eng_len = max_eng_len
        self.max_chn_len = max_chn_len
        self.eng_vocab, self.chn_vocab, self.eng_len, \
            self.chn_len, self.train_loader, self.test_loader = \
            gen_dict(SOS, EOS, PAD, UNK, START_INDEX, batch_size, eng_len, chn_len, 
                     min_freq=min_freq, max_vocab=max_vocab, device=device, 
                     max_eng_len=max_eng_len, max_chn_len=max_chn_len)
        self.eng_rev_map = {v: k for k, v in self.eng_vocab.items()}
        self.chn_rev_map = {v: k for k, v in self.chn_vocab.items()}
    
    def tokenize(self, sentence: str, lang: Enum):
        if lang == Language.ENGLISH:
            sentence = sentence.lower()[:-1].split() + list(sentence[-1])
            sentence = [self.eng_vocab[word] if word in self.eng_vocab else self.UNK for word in sentence]
            sentence = [self.eng_vocab["<sos>"]] + sentence + [self.eng_vocab["<eos>"]] + \
                ([self.eng_vocab["<pad>"]] * (self.eng_len - len(sentence) - 2))
            return torch.Tensor(sentence).long().to(self.device)
        if lang == Language.CHINESE:
            sentence = [self.chn_vocab[word] if word in self.chn_vocab else self.UNK for word in sentence]
            sentence = [self.chn_vocab["<sos>"]] + sentence + [self.chn_vocab["<eos>"]] + \
                ([self.chn_vocab["<pad>"]] * (self.chn_len - len(sentence) - 2))
            return torch.Tensor(sentence).long().to(self.device)
        raise ValueError("Invalid language")

    def untokenize(self, sentence: torch.Tensor, lang: Enum):
        if lang == Language.ENGLISH:
            res = ' '.join(self.eng_rev_map[i.item()] for i in sentence)
        elif lang == Language.CHINESE:
            res = ''.join(self.chn_rev_map[i.item()] for i in sentence)
        else:
            raise ValueError("Invalid language")
        res = res.split("<eos>")[0].removeprefix("<sos>").strip()
        if res[-1] in ['.', '?', '!'] and res[-2] == ' ':
            res = res[:-2] + res[-1]
        return res
    
    def rand_from_train(self):
        idx = torch.randint(0, len(self.train_loader.dataset), (1,)).item()
        eng, chn = self.train_loader.dataset[idx]
        return self.untokenize(eng, Language.ENGLISH), self.untokenize(chn, Language.CHINESE)
    
    def rand_from_test(self):
        idx = torch.randint(0, len(self.test_loader.dataset), (1,)).item()
        eng, chn = self.test_loader.dataset[idx][0]
        return self.untokenize(eng, Language.ENGLISH), self.untokenize(chn, Language.CHINESE)