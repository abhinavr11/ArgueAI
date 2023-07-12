import openai
import random

apikey= "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"

context=""

def generate_case():
    global context

    #content=["traffic", "crime", "environment", "family", "cybersecurity"]
    #select_content=content[random.randint(0,4)]
    select_content="traffic"
    prompt="Generate a simple case related to " + select_content + " of 2 or 3 lines and give only the case in double quotes and nothing else. Not a single word extra. Give only one case not multiple."

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
    #print('Case: ', context)
    print('Prosecutor: ', prompt)
    print(' ')

    prompt= "\"" + prompt + "\" \n" + "is this the correct rule that applies to the case " + "\"" + context + "\" \n" + "Answer in one word. 'Yes' or 'no' only. No extra word.\nBe lineant"
    # Set up your OpenAI API credentials
    openai.api_key = apikey

    # Define the model and parameters
    model = 'text-davinci-003'
    max_tokens = 10

    # Generate a response
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=0.1,
        top_p=0.8,
        frequency_penalty=1.0,
        presence_penalty=1.0
    )
    #print('yup8')
    # Extract the generated reply
    reply = str(response.choices[0].text.strip())
    #print('yup9')

    if "yes" in reply.lower():
        reward=10
        print('REWARDED \n')
    else:
        reward=-1
        print('penalized \n')

    return reward

def reward_defence(prompt):
    print('Defence: ', prompt)

    prompt= "\"" + prompt + "\" \n" + "is this a correct statement that can be told in defence to the case " + "\"" + context + "\" \n" + "Answer in one word. 'Yes' or 'no' only. No extra word. \nBe lineant"
    # Set up your OpenAI API credentials
    openai.api_key = apikey

    # Define the model and parameters
    model = 'text-davinci-003'
    max_tokens = 10

    # Generate a response
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=0.1,
        top_p=0.8,
        frequency_penalty=1.0,
        presence_penalty=1.0
    )

    # Extract the generated reply
    reply = str(response.choices[0].text.strip())

    if "yes" in reply.lower():
        reward=10
        print('\n REWARDED \n')
    else:
        reward=-1
        print('\n penalized \n')

    return reward

'''# Example usage
prompt = "What is the capital of France?"
response = generate_response(prompt)
print(response)'''
