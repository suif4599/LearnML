import os
import torch
import warnings
from math import inf

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
    return eng_vocab, chn_vocab, eng_len - 1, chn_len - 1, train_loader, test_loader

def tokenize(sentence, UNK, eng_vocab, length, device=None): # eng->chn
    sentence = sentence.lower()[:-1].split() + list(sentence[-1])
    sentence = [eng_vocab[word] if word in eng_vocab else UNK for word in sentence]
    sentence = [eng_vocab["<sos>"]] + sentence + [eng_vocab["<eos>"]] + ([eng_vocab["<pad>"]] * (length - len(sentence) - 2))
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.Tensor(sentence).long().to(device)

    