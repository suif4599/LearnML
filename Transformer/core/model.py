import torch
import os
import warnings
import tqdm
from time import time
from datetime import datetime
from .layer import PositionalEncodingLayer, EncoderLayer, DecoderLayer
from .constants import *
from data import tokenize

SAMPLE = ["I am a student.",
          "I'd like to have a cup of coffee.",
          "I want to go to the library.",
          "Can you help me with my homework?",
          "I am not feeling well today.",
          "Tom gave me a book as a birthday present.",
          "I am looking forward to seeing you.",
          "I am sorry for being late.",
          "I am happy to see you again.",
          "I am so excited about the concert."]

class TransformerTranslator(torch.nn.Module):
    "Input: (batch_size, seq_len)"
    def __init__(self, source_vocab, target_vocab, 
                 source_seq_len, target_seq_len, 
                 d_model, n_head, n_layer, dropout=0.1):
        super().__init__()
        self.__source_vocab = source_vocab
        self.__target_vocab = target_vocab
        source_vocab_size = len(source_vocab)
        target_vocab_size = len(target_vocab)
        self.__source_seq_len = source_seq_len
        self.__target_seq_len = target_seq_len
        self.__d_model = d_model
        self.__n_head = n_head
        self.__n_layer = n_layer
        self.__dropout = dropout
        self.encoder_embedding = torch.nn.Embedding(source_vocab_size, d_model)
        self.encoder_pos_encoder = PositionalEncodingLayer(d_model, source_seq_len)
        self.encoder_layers = torch.nn.ModuleList([EncoderLayer(d_model, n_head, dropout=dropout) for _ in range(n_layer)])
        self.decoder_embedding = torch.nn.Embedding(target_vocab_size, d_model)
        self.decoder_pos_encoder = PositionalEncodingLayer(d_model, target_seq_len)
        self.decoder_layers = torch.nn.ModuleList([DecoderLayer(d_model, n_head, target_seq_len, dropout=dropout) for _ in range(n_layer)])
        self.output_linear = torch.nn.Linear(d_model, target_vocab_size, bias=False)
    
    def forward(self, _input, _output):
        input_padding_mask = (_input == PAD)
        output_padding_mask = (_output == PAD)
        _input = self.encoder_embedding(_input)
        _input = self.encoder_pos_encoder(_input)
        for encoder_layer in self.encoder_layers:
            _input = encoder_layer(_input, input_padding_mask)
        _output = self.decoder_embedding(_output)
        _output = self.decoder_pos_encoder(_output)
        for decoder_layer in self.decoder_layers:
            _output = decoder_layer(_output, _input, input_padding_mask, output_padding_mask)
        _output = self.output_linear(_output)
        return _output
    
    def start_training(self, train_loader, test_loader, epochs, optimizer, 
                       device=None, save_path=None, only_save_params=False):
        if save_path is None:
            warnings.warn("No save path provided, model will not be saved.")
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self = self.to(device)
        print(f"Training on {device}")
        for epoch in range(epochs):
            self.train()
            t = time()
            train_loss = 0
            for _input, _output in tqdm.tqdm(train_loader, total=len(train_loader)):
                optimizer.zero_grad()
                output = self(_input[:, :-1], _output[:, :-1])
                loss = loss_fn(output.view(-1, output.size(-1)), _output[:, 1:].contiguous().view(-1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            self.eval()
            t_train = time() - t
            train_loss = train_loss / len(train_loader)
            with torch.no_grad():
                total_loss = 0
                for _input, _output in test_loader:
                    output = self(_input[:, :-1], _output[:, :-1])
                    loss = loss_fn(output.view(-1, output.size(-1)), _output[:, 1:].contiguous().view(-1))
                    total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(test_loader):.2f}, Train Loss: {train_loss:.2f}")
            t_test = time() - t_train - t
            for s in SAMPLE:
                print(f"{s} -> {self.translate(s)}")
            print(f"Time: {t_train:.2f}s (train), {t_test:.2f}s (test)")
            if save_path is not None:
                if only_save_params:
                    torch.save(self.state_dict(), os.path.join(save_path, f"model_{datetime.now().strftime('%Y%m%d%H%M%S')}_epoch{epoch + 1}.pt"))
                else:
                    torch.save(self, os.path.join(save_path, f"model_{datetime.now().strftime('%Y%m%d%H%M%S')}_epoch{epoch + 1}.pth"))
        return self
    
    @classmethod
    def load(cls, file, *args, **kwargs):
        if os.path.splitext(file)[-1] == ".pth":
            return torch.load(file)
        if os.path.splitext(file)[-1] == ".pt":
            model = cls(*args, **kwargs)
            model.load_state_dict(torch.load(file, weights_only=True))
            return model
        raise ValueError("Invalid file format")
    
    def __translate(self, sentence, device=None):
        if not isinstance(sentence, torch.Tensor):
            sentence = torch.tensor(sentence).long()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()
        self = self.to(device)
        sentence = sentence.unsqueeze(0).to(device)
        output = [SOS] + [UNK] * (self.__target_seq_len - 1)
        output = torch.tensor(output).long().to(device).unsqueeze(0)
        for i in range(self.__target_seq_len - 1):
            _output = self(sentence, output)
            _output = _output.argmax(dim=-1)
            if _output[:, i].item() == EOS:
                output[:, i + 1] = EOS
                break
            output[:, i + 1] = _output[:, i].item()
        return output.squeeze(0).cpu().numpy()
    
    def translate(self, src):
        tar_rev_map = {v: k for k, v in self.__target_vocab.items()}
        src = src.strip()
        if not src.endswith(".") or not src.endswith("?") or not src.endswith("!"):
            src += "."
        tar = ''.join(tar_rev_map[i] for i in self.__translate(tokenize(src, UNK, self.__source_vocab, self.__source_seq_len)))
        tar = tar.split("<eos>")[0].removeprefix("<sos>")
        return tar