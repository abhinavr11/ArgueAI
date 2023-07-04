from Prosecutor_final import *
from env_final import *
from Defence_final import *
import torch

env=CourtRoomEnvironment()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_actions = len(env.action_space)

prosecutor=Prosecutor_Agent(n_actions)
defence=Defence_Agent(n_actions)

state=input('Enter case contexts:')
context=input('Enter the extra case contexts for the defence')
state=env.Tokenize(state)
context=env.Tokenize(context)

prosecutor_action=prosecutor.policy_net(state).max(1)[1].view(1, 1)
print('prosecutor:', corpus)

new_state=torch.cat(state,context,env.corpus_rules[prosecutor_action])

defence_action=defence.policy_net(new_state).max(1)[1].view(1, 1)
