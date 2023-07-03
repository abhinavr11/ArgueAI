import random
import corpus_data 
import torch
from transformers import BertTokenizer, BertModel

class CourtRoomEnvironment:
    def __init__(self):
        self.corpus_rules = corpus_data.rule_list
        self.action_space=[i for i in range(len(self.corpus_rules))]

    def Tokenize(self, state):
        # Tokenize the prompt with bert
        # Load pre-trained BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Load pre-trained BERT model
        model = BertModel.from_pretrained('bert-base-uncased')

        # Tokenize the input text
        tokens = tokenizer.tokenize(state)

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

        return encoded_tensor
    
    def reset(self):
        case_context= input('Enter the case context')
        defence_context=input('Enter the defence context')
        
        case_context=self.Tokenize(case_context)
        defence_context=self.Tokenize(defence_context)

        return case_context, defence_context 
    
    def step(self, argument):
        print(self.corpus_rules[argument])
        #incorporate showing relevant argument in prosecutor/defence

        reward=0.0 #incorporate RLHF

        done=True if reward>=10 else False #flagged by a high reward

        return reward, done