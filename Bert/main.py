from bertviz import head_view, model_view
from transformers import BertTokenizer, BertModel
import os
import torch
model_version = 'bert-base-uncased'
path = os.path.join(os.path.dirname(__file__), model_version)
if os.path.exists(path):
    print("Loading model from disk")
    model = BertModel.from_pretrained(path, output_attentions=True)
else:
    print("Downloading model")
    model = BertModel.from_pretrained("bert-base-uncased", torch_dtype=torch.float16, attn_implementation="sdpa")
    model.save_pretrained('bert-base-uncased')
print("Model loaded")
tokenizer = BertTokenizer.from_pretrained(model_version)
sentence_a = "The cat sat on the mat"
sentence_b = "The cat lay on the rug"
inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt')
input_ids = inputs['input_ids']
token_type_ids = inputs['token_type_ids']
attention = model(input_ids, token_type_ids=token_type_ids)[-1]
sentence_b_start = token_type_ids[0].tolist().index(1)
input_id_list = input_ids[0].tolist() # Batch index 0
tokens = tokenizer.convert_ids_to_tokens(input_id_list) 
head_view(attention, tokens)