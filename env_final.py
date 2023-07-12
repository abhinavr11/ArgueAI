import random
import corpus_data 
import torch
from transformers import BertTokenizer, BertModel
import agent_training.training_using_gpt_api as api
from sentence_transformers import SentenceTransformer
import openai
import time

import agent_training.prompting as api2

model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

MAX_LENGTH = 128

class CourtRoomEnvironment:
    def __init__(self):
        self.corpus_rules = corpus_data.rule_list
        self.action_space=[i for i in range(len(self.corpus_rules))]
    
    def tokenize_text(self, text):
        tokens = tokenizer.tokenize(text)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        return tokens
    
    def pad_sequences(self, token_ids):
        if len(token_ids) > MAX_LENGTH:
            token_ids = token_ids[:MAX_LENGTH]
        else:
            token_ids = token_ids + [0] * (MAX_LENGTH - len(token_ids))
        return token_ids
    
    def extract_bert_embeddings(self, input_tensors):
        with torch.no_grad():
            model.eval()
            outputs = model(input_tensors)
            embeddings = outputs[0][0]
            #print(embeddings.shape)
        return embeddings

    def Tokenize(self, state):
        # Tokenize the prompt with bert
        sentence= state

        '''tokens= self.tokenize_text(sentence)
        token_ids= tokenizer.convert_tokens_to_ids(tokens)

        padded_token_ids = self.pad_sequences(token_ids)
        input_tensors= torch.tensor([padded_token_ids])
        embeddings= self.extract_bert_embeddings(input_tensors)

        embeddings= embeddings.view(1, -1)'''

        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeddings = model.encode(sentence)
        embeddings=torch.tensor(embeddings)
        embeddings=embeddings.view(1,-1)
        #print(embeddings.shape)

        return embeddings
    
    def reset(self):
        #case_context= input('Enter the case context: ')
        #defence_context=input('Enter the defence context: ')

        '''try:
            case_context=api.generate_case()
        except openai.error.RateLimitError:
            time.sleep(1)
            case_context=api.generate_case()'''
        
        case_context= api2.generate_case()

        print('Case generated: ', case_context)
        #print('hello')
        #defence_context="Case Context: " + case_context+ "-- Defence Context: " +defence_context

        case_context=self.Tokenize(case_context)
        #defence_context=self.Tokenize(defence_context)
        defence_context= case_context

        #print(case_context.shape)

        return case_context, defence_context 
    
    def step(self, argument, defence_argument):
        '''print(' ')
        print('Prosecutor: ', self.corpus_rules[argument])
        print('Defence: ', self.corpus_rules[defence_argument])
        print(' ')
        #incorporate showing relevant argument in prosecutor/defence in server

        #incorporate RLHF
        prosecutor_reward=0.0 
        defence_reward=0.0'''

        #done=False #if prosecutor_reward >= 10.0 & defence_reward >= 10.0 else False #flagged by a high reward
        '''if prosecutor_reward >= 10.0 and defence_reward >= 10.0:
            done =True
        else:
            done=False'''
        #print('yup7')
        
        '''try:
            prosecutor_reward= api.reward_prosecutor(self.corpus_rules[argument])
            #print('yup6')
            defence_reward= api.reward_defence(self.corpus_rules[defence_argument])
        except openai.error.RateLimitError:
            time.sleep(1)
            prosecutor_reward= api.reward_prosecutor(self.corpus_rules[argument])
            #print('yup6')
            defence_reward= api.reward_defence(self.corpus_rules[defence_argument])'''
        
        prosecutor_reward= api2.reward_prosecutor(self.corpus_rules[argument])
        defence_reward= api2.reward_defence(self.corpus_rules[defence_argument])
            
        #print('yup5')
        if int(prosecutor_reward) >= 10 and int(defence_reward) >= 10:
            done= True
        else:
            done = False
        
        return prosecutor_reward, defence_reward, done