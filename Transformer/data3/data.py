import torch
import os
import gc
import re
import pyarrow.parquet as parquet
import opencc
import pandas as pd
from math import inf
from enum import Enum

class Language(Enum):
    ENGLISH = 0
    CHINESE = 1

class TranslateDataset(torch.utils.data.Dataset):
    "set max_rows to -1 to load all data"
    LANGUAGE = Language
    def __init__(self, SOS=0, EOS=1, PAD=2, UNK=3, START_INDEX=4, *, 
                 batch_size=32, max_rows=200_000, 
                 min_eng_freq=None, min_chn_freq=None, 
                 max_eng_vocab=None, max_chn_vocab=None, device=None, 
                 max_eng_len=inf, max_chn_len=inf, 
                 train_frac=0.9):
        super().__init__()
        self.__metadata = {"SOS": SOS, "EOS": EOS, "PAD": PAD, "UNK": UNK, "START_INDEX": START_INDEX,
                           "batch_size": batch_size, "max_rows": max_rows,
                           "min_eng_freq": min_eng_freq, "min_chn_freq": min_chn_freq,
                           "max_eng_vocab": max_eng_vocab, "max_chn_vocab": max_chn_vocab,
                           "device": device, "max_eng_len": max_eng_len, "max_chn_len": max_chn_len,
                           "train_frac": train_frac}
        self.__init(SOS, EOS, PAD, UNK, START_INDEX, 
                    batch_size, max_rows, 
                    min_eng_freq, min_chn_freq, 
                    max_eng_vocab, max_chn_vocab, device, 
                    max_eng_len, max_chn_len, 
                    train_frac)
    
    def __init(self, SOS, EOS, PAD, UNK, START_INDEX, 
               batch_size, max_rows, 
               min_eng_freq, min_chn_freq, 
               max_eng_vocab, max_chn_vocab, device, 
               max_eng_len, max_chn_len, 
               train_frac, __vocab=None):
        max_eng_len += 2
        max_chn_len += 2
        self.SOS = SOS
        self.EOS = EOS
        self.PAD = PAD
        self.UNK = UNK
        self.START_INDEX = START_INDEX
        self.batch_size = batch_size
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        row = 0
        _data = pd.DataFrame(columns=["zh", "en"])
        print("loading data...")
        for i in range(28):
            path = os.path.join(os.path.dirname(__file__), "raw_data", f"train-{i:>05}-of-00028.parquet")
            if not os.path.exists(path):
                break
            data = parquet.ParquetDataset(path).read()
            row += data.num_rows
            _data = pd.concat([_data, data.to_pandas()])
            del data
            gc.collect()
            if row >= max_rows > 0:
                break
        self.data = _data.iloc[:max_rows]
        cc = opencc.OpenCC("t2s")
        self.data.loc[:, "zh"] = self.data.zh.apply(cc.convert)
        self.rows = len(self.data)
        self.train_frac = train_frac
        del _data
        gc.collect()
        print("tokenizing data...")
        if __vocab is None:
            eng_vocab_cnt = {}
            chn_vocab_cnt = {}
            for i in range(len(self.data)):
                for word in self.split_eng(self.data.iloc[i].en):
                    eng_vocab_cnt[word] = eng_vocab_cnt.get(word, 0) + 1
                for word in self.data.iloc[i].zh:
                    chn_vocab_cnt[word] = chn_vocab_cnt.get(word, 0) + 1
            if min_eng_freq is not None:
                if isinstance(min_eng_freq, float):
                    min_freq = int(min_eng_freq * self.rows)
                eng_vocab_cnt = {word: cnt for word, cnt in eng_vocab_cnt.items() if cnt >= min_freq}
            if min_chn_freq is not None:
                if isinstance(min_chn_freq, float):
                    min_freq = int(min_chn_freq * self.rows)
                    chn_vocab_cnt = {word: cnt for word, cnt in chn_vocab_cnt.items() if cnt >= min_freq}
            if max_eng_vocab is not None:
                eng_vocab_cnt = dict(sorted(eng_vocab_cnt.items(), key=lambda x: -x[1])[:max_eng_vocab])
            if max_chn_vocab is not None:
                chn_vocab_cnt = dict(sorted(chn_vocab_cnt.items(), key=lambda x: -x[1])[:max_chn_vocab])
            print(f"English filtered vocab size: {len(eng_vocab_cnt)}")
            print(f"Chinese filtered vocab size: {len(chn_vocab_cnt)}")
            self.eng_vocab = {word: i for i, word in enumerate(eng_vocab_cnt, START_INDEX)}
            self.chn_vocab = {word: i for i, word in enumerate(chn_vocab_cnt, START_INDEX)}
            self.eng_vocab["<sos>"] = self.chn_vocab["<sos>"] = SOS
            self.eng_vocab["<eos>"] = self.chn_vocab["<eos>"] = EOS
            self.eng_vocab["<pad>"] = self.chn_vocab["<pad>"] = PAD
            self.eng_vocab["<unk>"] = self.chn_vocab["<unk>"] = UNK
        else:
            self.eng_vocab, self.chn_vocab = __vocab
        self.eng_len = max(len(self.split_eng(self.data.iloc[i].en)) for i in range(len(self.data)))
        if self.eng_len > max_eng_len:
            self.eng_len = max_eng_len
            self.data = self.data[self.data.en.apply(lambda x: len(self.split_eng(x)) <= max_eng_len - 4)]
        self.chn_len = max(len(self.data.iloc[i].zh) for i in range(len(self.data)))
        if self.chn_len > max_chn_len:
            self.chn_len = max_chn_len
            self.data = self.data[self.data.zh.apply(lambda x: len(x) <= max_chn_len - 4)]
        self.eng_len -= 2
        self.chn_len -= 2
        self.eng_rev_map = {i: word for word, i in self.eng_vocab.items()}
        self.chn_rev_map = {i: word for word, i in self.chn_vocab.items()}
        print(f"English max length: {self.eng_len}")
        print(f"Chinese max length: {self.chn_len}")
        self.rows = len(self.data)
        print(f"Data size: {self.rows}")
        print("Generating dataset...")
        self.dataset = torch.utils.data.TensorDataset(
            torch.stack([self.tokenize(self.data.iloc[i].en, Language.ENGLISH) for i in range(self.rows)]),
            torch.stack([self.tokenize(self.data.iloc[i].zh, Language.CHINESE) for i in range(self.rows)])
        )
        self.train_size = int(self.rows * train_frac)
        self.test_size = self.rows - self.train_size
        self.train, self.test = torch.utils.data.random_split(self.dataset, [self.train_size, self.test_size])
        self.train_loader = torch.utils.data.DataLoader(self.train, batch_size=batch_size, shuffle=True, drop_last=False)
        self.test_loader = torch.utils.data.DataLoader(self.test, batch_size=batch_size, shuffle=True, drop_last=False)
        print("Data loaded")
    
    def split_eng(self, data: str, regex=re.compile(r"[\w]+\b|[.,!?;:]")):
        return re.findall(regex, data.lower())

    def tokenize(self, sentence: str, lang: Enum):
        if lang == Language.ENGLISH:
            sentence = self.split_eng(sentence)
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
        return res
    
    def rand_from_train(self):
        eng, chn = self.train_loader.dataset[torch.randint(0, self.train_size, (1,)).item()]
        return self.untokenize(eng, Language.ENGLISH), self.untokenize(chn, Language.CHINESE)
    
    def rand_from_test(self):
        eng, chn = self.test_loader.dataset[torch.randint(0, self.test_size, (1,)).item()]
        return self.untokenize(eng, Language.ENGLISH), self.untokenize(chn, Language.CHINESE)
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state["dataset"] = None
        state["train_loader"] = None
        state["test_loader"] = None
        state["train"] = None
        state["test"] = None
        return state
    
    def reload(self):
        self.__init(**self.__metadata, __vocab=(self.eng_vocab, self.chn_vocab))
        return self

# dataset = TranslateDataset(min_eng_freq=5e-5,
#                            max_eng_len=32, max_chn_len=64)
# for i in range(10):
#     loader = dataset.train_loader
#     print(dataset.untokenize(loader.dataset[i][0], Language.ENGLISH))
#     print(dataset.untokenize(loader.dataset[i][1], Language.CHINESE))
#     print()
