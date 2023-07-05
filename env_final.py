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
        case_context= input('Enter the case context: ')
        defence_context=input('Enter the defence context: ')
        defence_context=case_context+defence_context

        case_context=self.Tokenize(case_context)
        defence_context=self.Tokenize(defence_context)

        return case_context, defence_context 
    
    def step(self, argument, defence_argument):
        print(' ')
        print('Prosecutor: ', self.corpus_rules[argument])
        print('Defence: ', self.corpus_rules[defence_argument])
        print(' ')
        #incorporate showing relevant argument in prosecutor/defence in server

        #incorporate RLHF
        prosecutor_reward=0.0 
        defence_reward=0.0

        done=False #if prosecutor_reward >= 10.0 & defence_reward >= 10.0 else False #flagged by a high reward
        '''if prosecutor_reward >= 10.0 and defence_reward >= 10.0:
            done =True
        else:
            done=False'''
        return prosecutor_reward, defence_reward, done