# ArgueAI

1. The `env_final.py` is the code for custom environment made for ArgueAI
2. The `Prosecutor_final.py` and `Defence_final.py` are the codes for the prosecutor and defence RL Agents respectively
3. `corpus_data.py` file stores the legal rules in an array which acts as action space for the RL Agents
4. `train.py` file is the code for the training of ArgueAI
5. The `agent_training` folder has the file `training_using_gpt_api.py` which simulates RLHF using openAI's 'text-davinci-003' model, which is used to train ArgueAI
6. The `memory` folder stores all the parameters of the agent after training
