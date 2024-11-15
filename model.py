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
    direction = "Output only python code. Import the data as a dataframe from MyData.csv. Write the result of the Python code to a csv file called data_did.csv. "
    query = direction + user_input
    #mydata = pd.read_csv('1000 rows ev.csv').to_string()

    response = client.chat(model='llama3.2', messages=[
    {
        'role': 'user',
        'content': query,
        'context': mydata,
    },
    ])
    print(response['message']['content'])



