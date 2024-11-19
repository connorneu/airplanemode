import os
import pandas as pd
import io
import csv
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


def run_model(client, user_input, mydata):
    input_path = user_input['import_file']
    input_task = user_input['user_input']
    column_headers = user_input['column_headers']
    print(len(mydata))
    query = f"""Write Python code which will answer the user's question.
                Read the the data as a DataFrame using {input_path}.
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
        'content': query
        #"options": {
        #    "num_ctx": 130000
        #}
    },
    ])
    code = response['message']['content'] 
    print('raw code')
    print(code)
    parsed_code = parse_code(code)
    print('praseed code')
    print(parsed_code)
    evaluate_code(parsed_code)


def parse_code(raw_code):
    c_code = raw_code.split('xXStartXx')[1]
    code = c_code.split('xXEndXx')[0].strip()
    return code


def evaluate_code(code):
    exec(code)


def run_model(client, user_input, mydata):
    input_path = user_input['import_file']
    input_task = user_input['user_input']
    column_headers = user_input['column_headers']
    print(len(mydata))
    query = f"""Write Python code which will answer the user's question.
                Read the the data as a DataFrame using {input_path}.
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
        'content': query
        #"options": {
        #    "num_ctx": 130000
        #}
    },
    ])
    code = response['message']['content'] 
    print('raw code')
    print(code)
    parsed_code = parse_code(code)
    print('praseed code')
    print(parsed_code)
    evaluate_code(parsed_code)
