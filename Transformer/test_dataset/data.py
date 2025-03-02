import torch
from core import *

def gen_data(batch_size=32, seq_len=64, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    # FROM vocab: lower letters
    # TO vocab: upper letters
    from_vocab = "abcdefghijklmnopqrstuvwxyz"
    to_vocab = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    from_vocab = {c: i for i, c in enumerate(from_vocab, START_INDEX)}
    to_vocab = {c: i for i, c in enumerate(to_vocab, START_INDEX)}
    to_vocab["<sos>"] = from_vocab["<sos>"] = SOS
    to_vocab["<eos>"] = from_vocab["<eos>"] = EOS
    to_vocab["<pad>"] = from_vocab["<pad>"] = PAD
    to_vocab["<unk>"] = from_vocab["<unk>"] = UNK
    # vocab_size: 26 + {START_INDEX}
    vocab_size = len(from_vocab)
    # cast mapping
    cast_mapping = {i: j for i, j in zip(range(START_INDEX, vocab_size), torch.randint(START_INDEX, vocab_size, (vocab_size - START_INDEX,)).tolist())}
    # FROM seq_len: Norm(30, 10)
    # TO seq_len: = FROM seq_len
    def gen_pair(target_len=seq_len):
        seq_len = int(torch.normal(30, 10, (1,)).item())
        seq_len = max(seq_len, 10)
        seq_len = min(seq_len, 50)
        _input = torch.randint(START_INDEX, vocab_size, (seq_len,)).tolist()
        _output = [cast_mapping[i] for i in _input]
        _input = [from_vocab["<sos>"]] + _input + [from_vocab["<eos>"]]
        _output = [to_vocab["<sos>"]] + _output[::-1] + [to_vocab["<eos>"]]
        _input += [from_vocab["<pad>"]] * (target_len - len(_input))
        _output += [to_vocab["<pad>"]] * (target_len - len(_output))
        return _input, _output
    def gen_batch(batch_size=batch_size):
        _input = []
        _output = []
        for _ in range(batch_size):
            _i, _o = gen_pair()
            _input.append(_i)
            _output.append(_o)
        return torch.tensor(_input), torch.tensor(_output)
    def gen_loader():
        return torch.utils.data.DataLoader(gen_batch(batch_size * 128), batch_size=batch_size)
    return gen_loader(), gen_loader(), vocab_size

