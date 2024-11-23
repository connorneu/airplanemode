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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
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


def suggest_actions(client, df, mydtypes):
    print('chunking data')
    docsplits = chunck_data(df)
    print('get embedding mode')
    embeddings = build_embedding_model()
    print('get llm')
    llm = build_llm()
    print('build retriever')
    retriever = build_retriever(docsplits, embeddings)
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know."
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
    results = rag_chain.invoke({"input": "What is the earliest vehicle model year?"})
    print(results)
    print()
    print()
    print(results['answer'])
    
    import sys
    sys.exit(1)