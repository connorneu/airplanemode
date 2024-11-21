import os
import pandas as pd
from ollama import Client
from langchain_core.vectorstores import InMemoryVectorStore


def create_partial_file(d, p, n):
    fp = os.path.join(d, p)
    with open(fp, 'rb') as input_file:
        head = [next(input_file) for _ in range(n)]
    for h in head:
        print(h.decode())
    tp = os.path.join(d, 'trunc.csv')
    with open(tp, 'w+') as text_file:
        for line in head:
            text_file.write(line.decode())  

def build_client():
    return Client()


def run_model(client, user_input, mydata, message_history):
    print()
    print('MESSAGE HISTORY')
    print(message_history)
    print()
    input_path = user_input['import_file']
    input_task = user_input['user_input']
    column_headers = user_input['column_headers']
    print('MY DATA')
    print(len(mydata))
    print(mydata)
    query = f"""Write Python code to accomplish the user_question using the user_data.
                The column headers in the code must be the same as in the data: {column_headers} 
                Read the the data as a pandas DataFrame using {input_path}.
                Write the result of the code as a DataFrame to a csv file and call it "doData_Output.csv". 
                Write "xXStartXx" at the start of the code and "xXEndXx" and the end of the code.
                Anything between xXStartXx and xXEndXx needs to be python code that can be fed directly to a compiler.
                user_question: {input_task}
                user_data: {mydata}
                
            """
    #This is the data that the code will be run on: {mydata}.

    response = client.chat(model='llama3.2', messages=[
    {
        'role': 'user',
        'content': query,
        'messages': message_history
        #"options": {
        #    "num_ctx": 130000
        #}
    },
    ])
    print('RESPONSE')
    print(response['message'])
    print()
    print('McResponse')
    print(response)
    code = response['message']['content'] 
    print('raw code')
    print(code)
    parsed_code = parse_code(code)
    print('praseed code')
    print(parsed_code)
    evaluate_code(parsed_code)
    message_history.append({"role": "user", "content": query})
    message_history.append({"role": "system", "content": code})
    
    return message_history


def parse_code(raw_code):
    c_code = raw_code.split('xXStartXx')[1]
    code = c_code.split('xXEndXx')[0].strip()
    return code


def evaluate_code(code):
    exec(code)


def suggest_actions(client, mydata, mydtypes):
    print("len suggestion data")
    print(len(mydata))
    print('numwords', len(mydata.split()))
    print()

    response = client.chat(model='llama3.2', messages=[
    {
        'role': 'user',
        'content': query,
    },
    ])
    suggestions = response['message']['content'] 
    print('suggestions')
    print(suggestions)
    return suggestions


crap = f"""You are making suggestions for how a user can analyze their data.
                Below are example suggestions. 
                Please format the output as a python list with 3 chosen suggestions exactly like this:
                ["SUGGESTION1", "SUGGESTION2", "SUGGESTION3"]
                Your ouput will be read by a Python interpreter directly so please don't output any additional text outside of the Python list.
                Examples:
                1. Calculate the difference between [DATE COLUMN 1] and [DATE COLUMN 2].
                2. Create a graph to visualize [NUMERICAL COLUMN] and [CATEGORICAL COLUMN].
                3. Create a graph to visualize [NUMERICAL COLUMN] over [DATE TIME COLUMN]
                3. Select every row which [CONDITION] in [COLUMN] and [CONDITION] in [COLUMN] when [CONDITION].
                4. Group the data by [CATEGORICAL COLUMN] and calculate the minimum, maximum, total, and average price based on [NUMERICAL COLUMN].
                5. [STRING OPERATION] on [OBJECT COLUMN]
                6. Change the format of [DATE COLUMN] to [NEW DATE FORMAT].
                8. Create a pivot table for [COLUMN 1] and [COLUMN 2].
                9. Classify each [STRING] as positive, neutral, or negative.
                Create a suggestion using these examples. The suggestion should be changed to be relevant to the user data.
                Use the column data types to ensure the suggestion is logical. 

            """