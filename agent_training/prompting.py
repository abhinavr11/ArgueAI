import requests
import random

context=""

def generate_case():
    global context

    #content=["traffic", "crime", "environment", "family", "cybersecurity"]
    #select_content=content[random.randint(0,4)]
    select_content="traffic"
    prompt="Generate a simple case related to " + select_content + " of 2 or 3 lines and give only the case in double quotes and nothing else. Strictly not a single word extra. Give only one case not multiple."

    url = 'https://api.catto.codes/v1/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer catto_key_3ZIXE75cxS0wgQ0aZ4E46402'
    }
    data = {
        'model': 'text-davinci-003',
        'prompt': prompt,
        'temperature': 0.7,
        'max_tokens': 100,
        'top_p': 1,
        'frequency_penalty': 0.0,
        'presence_penalty': 0.0,
        'stop': None
    }

    response = requests.post(url, headers=headers, json=data)

    for t in range(3):
        if response.status_code == 200:
            result = response.json()
            # Process the result
            #print(result)
            reply= str(result['choices'][0]['text'])
            context=reply

            return reply
        else:
            print('Request failed with status code:', response.status_code)
            continue


def reward_prosecutor(prompt):
    print('Prosecutor: ', prompt)
    print(' ')

    prompt= "\"" + prompt + "\" \n" + "is this the best rule that applies to the case " + context + "\n" + "Answer in one word. 'Yes' or 'no' only. No extra word.\nBe precise and accurate upto a decent level"

    url = 'https://api.catto.codes/v1/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer catto_key_3ZIXE75cxS0wgQ0aZ4E46402'
    }
    data = {
        'model': 'text-davinci-003',
        'prompt': prompt,
        'temperature': 0.5,
        'max_tokens': 10,
        'top_p': 0.8,
        'frequency_penalty': 1.0,
        'presence_penalty': 1.0,
        'stop': None
    }

    for t in range(3):
        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            result = response.json()
            # Process the result
            #print(result)
            reply= str(result['choices'][0]['text'])

            if "yes" in reply.lower():
                reward=10
                print('REWARDED \n')
            else:
                reward=-10
                print('penalized \n')

            return reward
        else:
            print('Request failed with status code:', response.status_code)
            continue

def reward_defence(prompt):
    print('Defence: ', prompt)
    print('')

    prompt="Assume a court scenario \nBen is accused in the case: "+ context + "\nStephen is a lawyer, trying to defend Ben in this case. Now Ben is thinking what to say. Can saying "+ "\"" + prompt + "\" save Ben?\nAssuming everything said is true \nBe lineant.\nSay 'Yes' or 'No' only. No extra word. Not a single extra. " #"\" \n" + "can this statement be told in a court by the defence lawyer to save the accused in the case " + context + " ?\n" + "Answer in one word. 'Yes' or 'no' only. No extra word. \nBe lineant."
    
    url = 'https://api.catto.codes/v1/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer catto_key_3ZIXE75cxS0wgQ0aZ4E46402'
    }
    data = {
        'model': 'text-davinci-003',
        'prompt': prompt,
        'temperature': 0.2,
        'max_tokens': 10,
        'top_p': 0.8,
        'frequency_penalty': 1.0,
        'presence_penalty': 1.0,
        'stop': None
    }

    for t in range(3):
        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            result = response.json()
            # Process the result
            #print(result)
            reply= str(result['choices'][0]['text'])

            if "yes" in reply.lower():
                reward=10
                print('REWARDED \n')
            else:
                reward=-10
                print('penalized \n')

            return reward
        else:
            print('Request failed with status code:', response.status_code)
            continue
