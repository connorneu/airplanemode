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
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains import create_history_aware_retriever

ui = None
llm = None
retriever = None
message_history = None


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


def run_model(user_input, llm, retriever, message_history, myui):
    global ui
    global llm_g
    global retriever_g
    global message_history_g
    ui = myui
    llm_g = llm
    retriever_g = retriever
    message_history_g = message_history


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


def is_python(line):
   try:
       ast.parse(line)
       return True
   except SyntaxError:
       return False


def extract_python_only(code):
    cleaned = ''
    lines = code.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        if i < len(lines) - 1:
            nextline = lines[i+1]
            print(nextline)
            if nextline.startswith('\t') or nextline.startswith('    '):
                sub = line + '\n'
                j = i + 1
                while j < len(lines)-1:
                    sub += lines[j] + '\n'
                    j += 1
                    nextline_j = lines[j]
                    if not nextline_j.startswith('\t') or nextline_j.startswith('    '):
                    #if not '\t' in nextline_j:
                        i = j
                        if is_python(sub):
                            cleaned += sub + '\n'
                        break
            else:
                if is_python(line):
                    cleaned += line + '\n'
        else:
            if is_python(line):
                cleaned += line + '\n'
        i += 1
    return cleaned


def parse_code(raw_code):
    c_code = raw_code.split('xXStartXx')[1]
    code = c_code.split('xXEndXx')[0].strip()
    return code


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


def rerun_after_error(code, error):
    system_prompt = (
        "Change the Python code so it does not generate any errors. "
        "Use the data provided as context. "
        "Here is the Python code: {code} "
        "Here is the error code it is generating: {error} "
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
            
        ]
    )
    model = OllamaLLM(model="llama3.2")
    question_answer_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(retriever_g, question_answer_chain)
    result = rag_chain.invoke({"input": "Rewrite the code to solve the error", "code": code, "error": error}) # , {"history": message_history}

    print('REULT')
    print(result)
    print("EROR REVISED CODE")
    code = result['answer']
    print(code)
    code = extract_python_only(code)
    print('code after cleanse')
    print(code)
    evaluate_code(code)


def evaluate_code(code):  # write methode to convert OBJ into non technical rrror to display to user
    try:                    # for example '<' not supported between instances of 'str' and 'int' converted to "This column is not a number"
        exec(code, globals())
        print('Code execution complete.')
    except Exception as e:
        print("CODE FAILURE")
        traceback.print_exc()
        #print()
        exc_type, exc_obj, exc_tb = sys.exc_info()
        #print('TYPE')
        #print(exc_type) # <class 'FileNotFoundError'>
        print("OBJ")
        print(exc_obj) # [Errno 2] No such file or directory: '/home/kman/VS_Code/pr...
        #print("TB")
        #print(exc_tb) # full error stack
        #print("RERUNNING WITHIN MODELS")
        rerun_after_error(code, exc_obj)


def generate_embeddingsQ(df):
    embeddings = OllamaEmbeddings(
        model="llama3",
    )
    text = df.stack().to_list()
    embeddings_df = embeddings.embed_documents(text)


def generate_embeddingsF(df):
    text_splitter = CharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    doc_splits = text_splitter.split_text(df.to_string())
    vectorstore = Chroma.from_texts(
        texts=doc_splits,
        collection_name="rag-chroma",
        embedding=embeddings.OllamaEmbeddings(model='nomic-embed-text')
    )
    retriever = vectorstore.as_retriever()
    return retriever


def chunck_data(df):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    #chunks = DataFrameLoader(df,
    #                        page_content_column='VIN (1-10)').load()
    data = df.to_string()
    chunks = text_splitter.split_text(data)
    docsplits = text_splitter.create_documents(chunks)
    return docsplits


def build_embedding_model():
    #embeddings = ollama.embeddings(
    #    model='nomic-embed-text'
    #)
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
    docsplits = chunck_data(df)
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
    return results['answer'], docsplits, embeddings, llm, retriever


def new_or_old(client, user_input):
    query = f"""
            You are a helpful assistant.
            A process has generated an output file.
            Your task is to determine if the user wants to undo the changes that were made or continue with the current output.
            Output True if the user wants to undo the changes that were made and output False if the user wants to continue with the current output.
            Your response needs to be exactly: True or False
            user response: {user_input}
            """
    response = client.chat(model='llama3.2', messages=[
    {
        'role': 'user',
        'content': query
    },
    ])
    is_redo = response['message']['content']
    print('ISREDO Response')
    print(is_redo)
    return is_redo