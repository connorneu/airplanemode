import os
import sys
import pandas as pd
import ollama
import traceback
import ast
from ollama import Client
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.chains import create_history_aware_retriever
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.prompts import PromptTemplate
import time


ui = None
llm_g = None
retriever = None
message_history_g = None

SYS_INSTRUCTIONS = ("""Please write Python code to analyze the user's data based on their description, using the provided dataset. 
            The dataset is located in a file named 'input_file.csv'. Follow these instructions carefully: 
            1. Read 'input_file.csv' into a pandas DataFrame named `data`.
            2. Analyze or manipulate the DataFrame based strictly on the user's instructions.
            3. Document the user's request in a comment at the start of the code to explain what the script does.
            4. Avoid using `print` statements or any direct console output in the code.
            5. Save the final output DataFrame as 'doData_Output.csv'. Ensure that it contains the correct results of the user's requested operation.
            6. Validate that the operations are performed correctly, and handle potential by raising an exception, such as missing columns or incorrect data types. 
            7. Pay close attention to which columns are relevant for the user's instructions
            Remember: The output must match the user's request exactly, and any deviations should be explained in comments."""
        )


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


def run_model_with_retreiver(user_input, llm, retriever, message_history, myui):
    global ui
    global llm_g
    global retriever_g
    #global message_history_g
    ui = myui
    #llm_g = llm
    retriever_g = retriever
    #message_history_g = message_history


    input_path = user_input['import_file']
    input_task = user_input['user_input']
    output_path = user_input['output_path']
    
    system_prompt = (
        "Write Python code to analyze the users data exactly as they describe using the provided context. "
        "All the data for the code is in a file called input_file.csv. "
        "Read input_file.csv into a DataFrame as the data for the code. "
        "Create Python code that does what the user asks. "
        "Write the result of the Python code to a DataFrame and export it as a csv called doData_Output.csv. "
        "Print one statement which explains the code you've generated. "
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
            
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    results = rag_chain.invoke({"input": input_task}) # , {"history": message_history}
    code = results['answer']
    print("RUN Model Result")
    print(results)
    print("RAW Code")
    print(code)
    message_history.append([HumanMessage(content=input_task), SystemMessage(content=code)])
    #code = parse_code(code)
    code = extract_python_only(code)
    print('code after cleanse')
    print(code)
    code = update_paths(code, input_path, output_path)
    code = find_print_line_commas(code)
    code = replace_prints(code)
    print('UPDATEDCODE')
    print(code)
    evaluate_code(code)
    return message_history, code


def update_prompt_with_history(message_history):
    prompt_messages = [
        *message_history,
        SystemMessagePromptTemplate.from_template("This is the DataFrame the user is analyzing: {dataset}"), #"Use these column headers when generating the Python code: {column_headers} \n 
        HumanMessagePromptTemplate.from_template("{input_task}"),
    ]
    return ChatPromptTemplate.from_messages(prompt_messages)


def run_model(user_input, llm, message_history, markdown_df, eval_attempts = 0):
    input_path = user_input['import_file']
    input_task = user_input['user_input']
    output_path = user_input['output_path']
    prompt = update_prompt_with_history(message_history)   

    # ! this is only because message_history is not propuerly setup in run_model()
    # ! this is to pass message history to analyze_user_prompt and then to evaluate_code
    message_history = prompt

    start_time = time.time()
    chain = prompt | llm
    print("---build chain %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    response = chain.invoke({"dataset": markdown_df, "input_task": input_task}) 
    print("---Invoke run_model %s seconds ---" % (time.time() - start_time))
    print("Response:")
    print(response)
    code = extract_python_only(response)
    print("input path", input_path)
    print("outputpath", output_path)
    code = update_paths(code, input_path, output_path)
    print(code)
    #with open('/home/kman/VS_Code/projects/AirplaneModeAI/a.py') as f:
    #    code = f.read()

    code = remove_elem(code, '#')
    code = remove_elem(code, 'input(', isreplace=True)


    code, message_history = analyze_user_prompt(input_task, code, llm, message_history, markdown_df)

    code = remove_main(code)
    print('UPDATEDCODE')
    print(code)
    
    evaluate_code(code, message_history, markdown_df, llm, input_task, eval_attempts)
    # message_history.append(AIMessage(content=code))
    print('run mode complete.')
    explanation = 'hey'
    return message_history, code, markdown_df, explanation


def remove_elem(code, elem, isreplace=False):
    clean = ''
    for line in code.split('\n'):
        if not line.strip().startswith(elem):
            clean += line + '\n'
        else:
            if isreplace:
                clean += 'pass' + '\n'
    return clean


def analyze_user_prompt(input_task, code, llm, message_history, markdown_df):
    #prompt_messages = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template("""
    #                Here is some Python code:
    #                {code}

    #                And here is what the user wants the Python code to do:
    #                {input_task}

    #                Does the code do everything that the user requires?
    #                """)])
    anal_prompt = SystemMessagePromptTemplate.from_template("""
                    Here is some Python code:
                    {code}

                    And here is what the user wants the Python code to do:
                    {input_task}

                    Does the code do everything that the user requires?
                    Ignore exception handling.
                    """)
    #!
    message_history.append(anal_prompt)
    #chain = prompt_messages | llm
    chain = message_history | llm
    response = chain.invoke({"dataset": markdown_df, "code": code, "input_task": input_task})
    print("DISCREPANCY ANALyzed")
    print(response)
    #prompt_messages.append(AIMessage(content=response))
    message_history.append(AIMessage(content=response))
    
    compare_prompt = SystemMessagePromptTemplate.from_template("""
                    Change the code, based on your analysis, to meet all the user's requirements.
                    Never change the file path or file names.
                    """)                  

    #prompt_messages.append(compare_prompt)
    #chain = prompt_messages | llm
    message_history.append(compare_prompt)
    chain = message_history | llm

    compare_response = chain.invoke({"dataset": markdown_df, "code": code, "input_task": input_task})
    print("NEW CODE - Errors fixed")
    print(compare_response)
    code = extract_python_only(compare_response)
    print('Update Code COde')
    print(code)
    return code, message_history


def is_python(line):
   try:
       if len(line.strip())>0:
           ast.parse(line)
           return True
       else:
           return False
   except SyntaxError:
       return False


def extract_python_only(code):
    cleaned = ''
    lines = code.split('\n')
    i = 0
    firstline = -1
    lastline = -1
    while i < len(lines):
        sub = ''
        line = lines[i]
        sub = line + '\n'
        if i < len(lines) - 1:
            j = i + 1
            while j < len(lines):
                sub += lines[j] + '\n'
                if is_python(sub):
                    firstline = i
                    lastline = j
                j += 1
            if lastline >= 0:
                break
        if firstline >= 0:
            lastline = firstline
            break
        i += 1
    cleaned = '\n'.join(lines[firstline:lastline + 1])
    return cleaned


def update_paths(code, input_path, output_path):
    code = code.replace('input_file.csv', input_path)
    code = code.replace('doData_Output.csv', output_path)
    return code


def find_print_line_commas(code):
    newcode = ''
    codelines = code.split('\n')
    for line in codelines:
        if 'print(' in line:
            newline = replace_print_commas(line)
            newcode += newline + '\n'
        else:
            newcode += line + '\n'
    return newcode


def replace_print_commas(s):
    isInside = False
    new_s = ''
    for i in range(len(s)):
        c = s[i]
        if c == '\'':
            if not isInside:
                isInside = True
            else:
                isInside = False
        if c == ',':
            if not isInside:
                new_s += '+' + '\' \'' + '+'
        else:
            new_s += c
    return new_s


def replace_prints(code):
    code = code.replace('print(', 'ui.ai_response(')
    return code


def rerun_after_error(code, error, message_history, markdown_df, llm, input_task, eval_attempts):
    #sys_message = [SystemMessagePromptTemplate.from_template(
    #    "The code is generating an error. Re-write the code so that it does not generate the error."
    #    "This is the code: {code}\n"
    #    "This is the error: {error}\n"
    #    "This is the data the code is written for: {dataset}"
    #)]
    sys_message = SystemMessagePromptTemplate.from_template(
        "The code is generating an error."
        "This is the code: {code}\n"
        "This is the error: {error}\n"
        "Change the code so that it does not generate an error."
    )
    message_history.append(sys_message)
    #prompt = ChatPromptTemplate.from_messages(sys_message)
    #prompt = sys_message
    chain = message_history | llm
    response = chain.invoke({"input_task": input_task, "code": code, "error": error, "dataset": markdown_df}) 
    print("RERUN _Response:")
    print(response)
    code = extract_python_only(response)
    print('codeonly -rerun')
    print(code)
    evaluate_code(code, message_history, markdown_df, llm, input_task, eval_attempts)


# need to remove if name equals main because of issues with exec
def remove_main(code):
    if 'if __name__' in code:
        print('cleaning main statement')
        clean = ''
        lines = code.split('\n')
        i = 0
        inMain = False
        while i < len(lines):
            if not inMain:
                if 'if __name__' in lines[i]:
                    inMain = True
                else:
                    clean += lines[i] + '\n'
            else:
                stripline = lines[i].strip()
                clean += stripline + '\n'
            i += 1
        return clean
    else:
        print('no main statement to clean')
        return code


# exec problems https://stackoverflow.com/questions/4484872/why-doesnt-exec-work-in-a-function-with-a-subfunction
def evaluate_code(code, message_history, markdown_df, llm, input_task, eval_attempts):  # write methode to convert OBJ into non technical rrror to display to user
    try:                   
        exec(code, None, globals())
        print('Code execution complete.')
    except Exception as e:
        print("CODE FAILURE")
        if eval_attempts < 0:
            return 'Failure. Nothing but failure.'
        eval_attempts += 1
        print('eval attempt:', eval_attempts)        
        trace = traceback.print_exc()
        print(trace)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("ErrorDetails_-__Subject")
        print(exc_obj) # [Errno 2] No such file or directory: '/home/kman/VS_Code/pr...
        if eval_attempts > 2:
            print("Failed 3 times. Restarting Process.")
            message_history = [SystemMessage(content=SYS_INSTRUCTIONS)]
            run_model(input_task, llm, message_history, markdown_df, eval_attempts = -1)
        else:
            rerun_after_error(code, trace, message_history, markdown_df, llm, input_task, eval_attempts)


def chunck_data(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(data)
    docsplits = text_splitter.create_documents(chunks)
    return docsplits


def build_embedding_model():
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
    )
    return embeddings


def build_llm():
    llm = OllamaLLM(model="llama3.2")
    return llm


def build_retriever(docsplits, embeddings):
    vectorstore = InMemoryVectorStore.from_documents(
        documents=docsplits, embedding=embeddings
    )
    retriever = vectorstore.as_retriever()
    return retriever


def suggest_actions(df):
    print('chunking data')
    md_df = df.to_markdown()
    docsplits = chunck_data(md_df)
    print('get embedding mode')
    embeddings = build_embedding_model()
    print('get llm')
    llm = build_llm()
    print('build retriever')
    retriever = build_retriever(docsplits, embeddings)
    system_prompt = (
        "You are an assistant for asking questions about data. "
        "Use the following pieces of retrieved context to generate "
        "questions that, if answered, would help "
        "improve understanding about the data. "
        "Format your answer as a python list so that each suggestion is an element in that list. "
        "Output only the suggestion list without any other statements outside the list or you will fail this task. "
        "The format must be exactly as follows inlcuding the quotes around the suggestions: ['SUGGESTION 1', 'SUGGESTION 2', 'SUGGESTION 3']"
        "\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    results = rag_chain.invoke({"input": "What are 3 questions we could create Python code to answer?"})
    print("Model Suggestions:")
    print(results['answer'])
    return results['answer'], docsplits, embeddings, llm, retriever, md_df