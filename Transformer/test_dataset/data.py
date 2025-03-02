import torch
from core import *

def gen_data(batch_size=32, seq_len=64, seed=None, device=None):
    seq_len += 1
    if seed is not None:
        torch.manual_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        return torch.utils.data.TensorDataset(torch.tensor(_input).to(device), \
            torch.tensor(_output).to(device))
    def gen_loader(scale):
        return torch.utils.data.DataLoader(gen_batch(scale * batch_size), batch_size=batch_size, shuffle=True)
    return gen_loader(128), gen_loader(16), vocab_size

