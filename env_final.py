import random
from corpus_data import *

class CourtRoomEnvironment:
    def __init__(self):
        self.corpus_rules = [] # import from corpus_data.py file as an array of tensors
        self.action_space=[i for i in range(len(self.corpus_rules))]

    def Tokenize(self, state):
        # Tokenize the prompt with bert
        return state
    