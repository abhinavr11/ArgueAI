import openai
import random

apikey= "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"

context=""

def generate_case():
    global context

    content=["traffic", "crime", "environment", "family", "cybersecurity"]
    select_content=content[random.randint(0,4)]
    prompt="Generate a random simple case related to " + select_content + " of 2-3 lines and give only the case in double quotes and nothing else. Not a single word extra"

    # Set up your OpenAI API credentials
    openai.api_key = apikey

    # Define the model and parameters
    model = 'text-davinci-003'
    max_tokens = 100

    # Generate a response
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    # Extract the generated reply
    reply = str(response.choices[0].text.strip())

    context=reply

    return reply

def reward_prosecutor(prompt):
    prompt= "\"" + prompt + "\" \n" + "is this the correct rule that applies to the case " + "\"" + context + "\" \n" + "Answer in one word. Yes or no only. No extra word"
    # Set up your OpenAI API credentials
    openai.api_key = apikey

    # Define the model and parameters
    model = 'text-davinci-003'
    max_tokens = 100

    # Generate a response
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    #print('yup8')
    # Extract the generated reply
    reply = str(response.choices[0].text.strip())
    #print('yup9')
    if "yes" in reply.lower():
        reward=10
        print('PROSECUTOR REWARDED')
    else:
        reward=-1
        print('prosecutor penalized')

    return reward

def reward_defence(prompt):
    prompt= "\"" + prompt + "\" \n" + "is it a correct defence that applies to the case " + "\"" + context + "\" \n" + "Answer in one word. Yes or no only. No extra word"
    # Set up your OpenAI API credentials
    openai.api_key = apikey

    # Define the model and parameters
    model = 'text-davinci-003'
    max_tokens = 100

    # Generate a response
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    # Extract the generated reply
    reply = str(response.choices[0].text.strip())

    if "yes" in reply.lower():
        reward=10
        print('DEFENCE REWARDED')
    else:
        reward=-1
        print('defence penalized')

    return reward

'''# Example usage
prompt = "What is the capital of France?"
response = generate_response(prompt)
print(response)'''
