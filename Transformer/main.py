import os
import torch
from core import *
import traceback
from tools.folder_visualize import visualize_folder
print(f"cuda is {'available' if torch.cuda.is_available() else 'not available'}")

from data import TranslateDataset, Language
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--email", nargs=2, help="Email host and password")
parser.add_argument("--no-io", action="store_true", help="Disable 0, 1, 2 handle")
args = parser.parse_args()
if args.email is not None:
    email_host, email_password = args.email
    enable_email = True
elif not args.no_io:
    email_host = input("Email host (press enter to skip): ").strip()
    email_password = None
    if email_host:
        email_password = input("Email password: ").strip()
        enable_email = True
    else:
        enable_email = False


if enable_email:
    mail_server = MailServer(email_host, email_password)
    mail_server.mainloop(delay=1, create_thread=True)
    def show(_):
        folder = os.path.join(os.path.dirname(__file__), "model_save")
        return visualize_folder(folder)
    print("Mail server started")
    def delete(content: str):
        print(content)
        for folder in os.listdir(os.path.join(os.path.dirname(__file__), "model_save")):
            if not os.path.isdir(os.path.join(os.path.dirname(__file__), "model_save", folder)):
                continue
            if content.strip().lower() == folder.lower():
                full_path = os.path.join(os.path.dirname(__file__), "model_save", folder)
                os.system(f"rm -rf {full_path}")
                return f"Model {folder} deleted"
        return f"Model {content} not found"
    mail_server.register("show", show, "Show all models saved")
    mail_server.register("delete", delete, "Delete a model")
    mail_server.send("Info", "main.py started")

# dataset = TranslateDataset(SOS, EOS, PAD, UNK, START_INDEX, 
#                            batch_size=32, min_freq=2, 
#                            max_eng_len=10, max_chn_len=15)
                

filename = os.path.join(os.path.dirname(__file__), "model_save", "TransformerTranslator_20250307145944")

warmup_epoch = 8
lr_max = 1e-4
lr_min = 1e-5

# model = TransformerTranslator.load(filename)
# model.dataset.reload()

try:
    if args.no_io:
        import sys
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
    dataset = TranslateDataset(max_rows=10_000, min_eng_freq=5e-5,
                            max_eng_len=64, max_chn_len=96,
                            batch_size=64, max_eng_vocab=10_000)
    model = TransformerTranslator(dataset, 256, 8, 8, dropout=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_min)
    lambda_lr = torch.optim.lr_scheduler.LambdaLR(optimizer, 
                                                lambda epoch: lr_max / lr_min if epoch >= warmup_epoch else (lr_max / lr_min) ** (epoch / warmup_epoch))
    model.start_training(10, optimizer, lambda_lr,
                        save_path=os.path.join(os.path.dirname(__file__), "model_save"),
                        save_each_num_epoch=3,
                        mail_server=mail_server)
    if args.no_io:
        mail_server.send("Info", "Training finished")
        exit()
    while 1:
        model.eval()
        eng = input("Input English sentence: ").strip()
        if eng == "exit":
            break
        print("Chinese translation:", model.translate(eng))
        print()
except Exception as e:
    info = traceback.format_exc()
    if enable_email:
        mail_server.send("Error", info)
    print(info)
    raise e

