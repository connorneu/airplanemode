import os
import pandas as pd
import io
import csv
from ollama import Client


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


def run_model(user_input, client, mydata):
    print('USER INPUT')
    print(user_input)
    input_path = user_input['import_file']
    input_task = user_input['user_input']
    column_headers = user_input['column_headers']
    print(input_path)
    print(input_task)
    query = f"""Write code to accomplish this task: {input_task}.
                Use the column headers provided: {column_headers}.
                Use the {input_path} as the dataset and read it as a DataFrame using it's full name.
                Write the result of the code as a DataFrame to a csv file and call it "doData_Output.csv". 
                Write "xXStartXx" at the start of the code and "xXEndXx" and the end of the code.
                Anything between xXStartXx and xXEndXx needs to be python code that can be fed directly to a compiler.
                Column headers are case sensitive.
            """
    response = client.chat(model='llama3.2', messages=[
    {
        'role': 'user',
        'content': query,
        'context': mydata,
    },
    ])
    code = response['message']['content'] 
    print('raw code')
    print(code)
    parsed_code = parse_code(code)
    print('praseed code')
    print(parsed_code)
    evaluate_codae(parsed_code)


def parse_code(raw_code):
    c_code = raw_code.split('xXStartXx')[1]
    code = c_code.split('xXEndXx')[0].strip()
    return code

def evaluate_codae(code):
    exec(code)


