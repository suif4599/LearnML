import torch
from .layer import PositionalEncodingLayer, EncoderLayer, DecoderLayer
from .constants import *
from .train import train_model

class TransformerTranslator(torch.nn.Module):
    "Input: (batch_size, seq_len)"
    def __init__(self, vocab_size, seq_len, d_model, n_head, n_layer):
        super().__init__()
        self.__vocab_size = vocab_size
        self.__seq_len = seq_len
        self.__d_model = d_model
        self.__n_head = n_head
        self.__n_layer = n_layer
        self.encoder_embedding = torch.nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncodingLayer(d_model, seq_len)
        self.encoder_layers = [EncoderLayer(d_model, n_head) for _ in range(n_layer)]
        self.decoder_embedding = torch.nn.Embedding(vocab_size, d_model)
        self.decoder_layers = [DecoderLayer(d_model, n_head, seq_len) for _ in range(n_layer)]
        self.output_linear = torch.nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, _input, _output):
        padding_mask = (_input == PAD)
        _input = self.encoder_embedding(_input)
        _input = self.pos_encoder(_input)
        for encoder_layer in self.encoder_layers:
            _input = encoder_layer(_input, padding_mask)
        _output = self.decoder_embedding(_output)
        for decoder_layer in self.decoder_layers:
            _output = decoder_layer(_output, _input, padding_mask)
        return _output
    
    def start_training(self, train_loader, test_loader, epochs, optimizer, device=None):
        print("Start training")
        return train_model(train_loader, test_loader, self, optimizer, torch.nn.CrossEntropyLoss(), epochs, device)