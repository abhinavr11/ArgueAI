import torch
from transformers import BertTokenizer, BertModel

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained BERT model
model = BertModel.from_pretrained('bert-base-uncased')

# Get user input
input_text = input("Enter a string: ")

# Tokenize the input text
tokens = tokenizer.tokenize(input_text)

# Add [CLS] and [SEP] tokens to the input tokens
tokens = ['[CLS]'] + tokens + ['[SEP]']

# Convert tokens to token IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)

# Convert token IDs to tensors
input_tensor = torch.tensor([token_ids])

# Pass the input tensor through the BERT model
output = model(input_tensor)

# Access the encoded representation (last hidden state)
encoded_tensor = output.last_hidden_state

# Print the encoded tensor
print(encoded_tensor)