import os
import pandas as pd
import ollama
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


def run_modelF(client, user_input, df, message_history):
    print()
    print('MESSAGE HISTORY')
    print(message_history)
    print()
    input_path = user_input['import_file']
    input_task = user_input['user_input']
    column_headers = user_input['column_headers']
    print('Input Task:', input_task)
    print('MY DATA')
    df = df.head(10)
    mydata = df.to_string()
    print(len(mydata))
    print(mydata)
    query = """
            You are a helpful assistant who generates Python code. 
            Write Python code to answer the question using the data provided. 
            The column headers in the code must be the same as in the data: {column_headers} 
            Read the data as a pandas DataFrame using {input_path}. 
            Write the result of the code as a DataFrame to a csv file and call it doData_Output.csv. 
            Write xXStartXx at the start of the code and xXEndXx and the end of the code. 
            Anything between xXStartXx and xXEndXx needs to be python code that can be fed directly to a compiler. 
            \n
            question: {input_task}
            \n
            data: {mydata}
            """
    print("QUERY")
    print(query)
    print()
    response = client.chat(model='llama3.2', messages=[
    {
        'role': 'user',
        'content': query
        #'messages': message_history
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


def run_model(user_input, llm, retriever):
    input_path = user_input['import_file']
    input_task = user_input['user_input']
    system_prompt = (
        "You are a helpful assistant who generates Python code. "
        "Write Python code to answer the question using the data provided. "
        "Read the data as a pandas DataFrame using input_file.csv. "
        "Write the result of the code as a DataFrame to a csv file and call it doData_Output.csv. "
        "Write xXStartXx at the start of the code and xXEndXx and the end of the code. "
        "Anything between xXStartXx and xXEndXx needs to be python code that can be fed directly to a compiler. "
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
    results = rag_chain.invoke({"input": input_task})
    code = results['answer']
    print("RUN Model Result")
    print(results)
    print()
    print()
    print(code)
    code = parse_code(code)
    print('praseed code')
    print(code)
    code = update_input_path(code, input_path)  
    evaluate_code(code)


def parse_code(raw_code):
    c_code = raw_code.split('xXStartXx')[1]
    code = c_code.split('xXEndXx')[0].strip()
    return code


def update_input_path(code, input_path):
    code = code.replace('input_file.csv', input_path)
    return code


def evaluate_code(code):
    exec(code)


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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
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
        "Format your answer as a python list so that each suggestion is an element in that list like this: ['SUGGESTION 1', 'SUGGESTION 2', 'SUGGESTION 3']"
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
    print(results)
    print()
    print()
    print(results['answer'])
    return results['answer'], docsplits, embeddings, llm, retriever


def new_or_old(client, user_input):
    query = f"""
            You are a helpful assistant.
            A process generated an output file.
            Your task is to determine if the user wants to undo the changes that were made or continue with the current output.
            Output True if the user wants to undo the changes that were made and output False if the user wants to continue with the current output.
            Do not ouptu anything else.
            user response: {user_input}
            """
    response = client.chat(model='llama3.2', messages=[
    {
        'role': 'user',
        'content': query
    },
    ])
    print('RESPONSE')
    print(response['message'])
    print()
    print('McResponse')
    print(response)
    return response