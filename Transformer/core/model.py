import torch
import os
import pickle
import warnings
import tqdm
import json
from time import time
from datetime import datetime
from .layer import PositionalEncodingLayer, EncoderLayer, DecoderLayer
from .constants import *
from data import TranslateDataset
Language = TranslateDataset.LANGUAGE

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
    def __init__(self, dataset: TranslateDataset, 
                 d_model, n_head, n_layer, dropout=0.1):
        super().__init__()
        global Language
        Language = dataset.LANGUAGE
        self.dataset = dataset
        self.__source_vocab = dataset.eng_vocab
        self.__target_vocab = dataset.chn_vocab
        source_vocab_size = len(dataset.eng_vocab)
        target_vocab_size = len(dataset.chn_vocab)
        self.__source_seq_len = dataset.eng_len - 1
        self.__target_seq_len = dataset.chn_len - 1
        self.__d_model = d_model
        self.__n_head = n_head
        self.__n_layer = n_layer
        self.__dropout = dropout
        self.encoder_embedding = torch.nn.Embedding(source_vocab_size, d_model)
        self.encoder_pos_encoder = PositionalEncodingLayer(d_model, self.__source_seq_len)
        self.encoder_layers = torch.nn.ModuleList([EncoderLayer(d_model, n_head, dropout=dropout) for _ in range(n_layer)])
        self.decoder_embedding = torch.nn.Embedding(target_vocab_size, d_model)
        self.decoder_pos_encoder = PositionalEncodingLayer(d_model, self.__target_seq_len)
        self.decoder_layers = torch.nn.ModuleList([DecoderLayer(d_model, n_head, self.__target_seq_len, dropout=dropout) for _ in range(n_layer)])
        self.output_linear = torch.nn.Linear(d_model, target_vocab_size, bias=False)
        self.__metadata = {"d_model": d_model, "n_head": n_head, "n_layer": n_layer, "dropout": dropout}
        self.__train_log = {}
    
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
    
    def start_training(self, epochs, optimizer, lr_scheduler=None,
                       device=None, save_path=None, model_name=None, 
                       save_each_num_epoch=1):
        if save_path is None:
            warnings.warn("No save path provided, model will not be saved.")
        train_loader = self.dataset.train_loader
        test_loader = self.dataset.test_loader
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self = self.to(device)
        print(f"Training on {device}")
        log = {}
        t = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.__train_log[f"train_{t}"] = log
        log["device"] = str(device)
        for epoch in range(epochs):
            self.train()
            t = time()
            train_loss = 0
            data = {"epoch": epoch + 1, 
                    "start_time": datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
                    "lr": optimizer.param_groups[0]["lr"]
                    }
            for _input, _output in tqdm.tqdm(train_loader, total=len(train_loader)):
                optimizer.zero_grad()
                output = self(_input[:, :-1], _output[:, :-1])
                loss = loss_fn(output.view(-1, output.size(-1)), _output[:, 1:].contiguous().view(-1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            if lr_scheduler is not None:
                lr_scheduler.step()
            self.eval()
            t_train = time() - t
            train_loss = train_loss / len(train_loader)
            data["train_loss"] = train_loss
            data["train_duration"] = t_train
            with torch.no_grad():
                total_loss = 0
                for _input, _output in test_loader:
                    output = self(_input[:, :-1], _output[:, :-1])
                    loss = loss_fn(output.view(-1, output.size(-1)), _output[:, 1:].contiguous().view(-1))
                    total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(test_loader):.2f}, Train Loss: {train_loss:.2f}")
            t_test = time() - t_train - t
            data["test_loss"] = total_loss / len(test_loader)
            data["test_duration"] = t_test
            log[f"epoch_{epoch + 1}"] = data
            print("Sample translations:")
            for s in SAMPLE:
                print(f"{s} -> {self.translate(s)}")
            print("Train translation:")
            src = self.dataset.rand_from_train()
            print(f"{src[0]} -> {self.translate(src[0])}")
            print(f"Time: {t_train:.2f}s (train), {t_test:.2f}s (test)")
            if (epoch + 1) % save_each_num_epoch == 0 and save_path is not None:
                if model_name is None:
                    self.save(save_path)
                else:
                    self.save(save_path, model_name)
        return self
    
    @classmethod
    def load(cls, folder: str):
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder {folder} does not exist")
        with open(os.path.join(folder, "metadata.json"), "r") as f:
            metadata = json.load(f)
        with open(os.path.join(folder, "train_log.json"), "r") as f:
            train_log = json.load(f)
        with open(os.path.join(folder, "dataset.pkl"), "rb") as f:
            dataset = pickle.load(f)
        model = cls(dataset, **metadata)
        model.load_state_dict(torch.load(os.path.join(folder, "model.pt"), weights_only=True))
        train_log["load_time"] = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        model.set_log(train_log)
        return model
    
    def set_log(self, log):
        self.__train_log = log

    def save(self, folder: str, model_name: str = "TransformerTranslator"):
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder {folder} does not exist")
        folder_name = os.path.join(folder, f"{model_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}")
        os.makedirs(folder_name)
        torch.save(self.state_dict(), os.path.join(folder_name, "model.pt"))
        with open(os.path.join(folder_name, "dataset.pkl"), "wb") as f:
            pickle.dump(self.dataset, f)
        with open(os.path.join(folder_name, "metadata.json"), "w") as f:
            json.dump(self.__metadata, f, indent=4, separators=(", ", ": "))
        with open(os.path.join(folder_name, "train_log.json"), "w") as f:
            json.dump(self.__train_log, f, indent=4, separators=(", ", ": "))
        return folder_name
        

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
            _output = self(sentence[:, :-1], output)
            _output = _output.argmax(dim=-1)
            if _output[:, i].item() == EOS:
                output[:, i + 1] = EOS
                break
            output[:, i + 1] = _output[:, i].item()
        return output.squeeze(0).cpu().numpy()
    
    def translate(self, src: str):
        src = src.strip()
        if not src.endswith(".") and not src.endswith("?") and not src.endswith("!"):
            src += "."
        tar = self.dataset.untokenize(self.__translate(self.dataset.tokenize(src, Language.ENGLISH)), Language.CHINESE)
        return tar