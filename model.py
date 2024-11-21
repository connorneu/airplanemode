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
    query = f"""You are making suggestions for how a user can analyze their data. Your output is just the suggestions without the original questions.
                Format the questions so that they don't use any technical terms and ask the question from the perspective of the user.
                Use the column_data_types provided to ensure that the suggestions make sense for the type of data in a particular column.
                Make a suggestion based off of the below examples but tailor them to the users data and its column data types. 
                You will make 3 total suggestions.
                Examples:
                Calculate the difference between arrival date and departure date.
                Graph the total sales by region.
                Select all the items which have total sales exceeding 10000, are manufactured in Canada, and were sold in the last 90 days. 
                Group the data by product type and calculate the minimum, maximum, total, and average price.
                Replace all the dashes in the phone number with dots.
                Change the Date Received data format to day/month/year.
                Split first and last name into two columns.
                Create a pivot table for each sales person and their sales.
                Classify each product review as positive, neutral, or negative.
                user_data: {mydata}
                column_data_types: {mydtypes}
                Output your result as a python list with each suggestion as an element in that list exactly like this:
                ["SUGGESTION1", "SUGGESTION2", "SUGGESTION3"]
            """
    response = client.chat(model='llama3.2', messages=[
    {
        'role': 'user',
        'content': query
        #"options": {
        #    "num_ctx": 130000
        #}
    },
    ])
    suggestions = response['message']['content'] 
    print('suggestions')
    print(suggestions)
    return suggestions
