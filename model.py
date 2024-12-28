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


def run_modelPP(user_input, llm, retriever, message_history, myui):
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
        HumanMessagePromptTemplate.from_template("{input}"),
        #SystemMessagePromptTemplate.from_template("Relevant information: {context}")
        #SystemMessagePromptTemplate.from_template("{dataset}")
    ]
    return ChatPromptTemplate.from_messages(prompt_messages)


def run_model(user_input, llm, retriever, message_history, myui, markdown_df, column_headers):
    global ui
    ui = myui
    input_path = user_input['import_file']
    input_task = user_input['user_input']
    output_path = user_input['output_path']
    #user_prompt = rewrite_user_prompt(input_task, llm)
    prompt = update_prompt_with_history(message_history)   

    start_time = time.time()
    chain = prompt | llm
    print("---build chain %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    response = chain.invoke({"dataset": markdown_df, "input": input_task}) 
    print("---Invoke run_model %s seconds ---" % (time.time() - start_time))
    print("Response:")
    print(response)
    code = extract_python_only(response)
    code = update_paths(code, input_path, output_path)

    #with open('/home/kman/VS_Code/projects/AirplaneModeAI/a.py') as f:
    #    code = f.read()

    code = remove_elem(code, '#')
    code = remove_elem(code, 'input(', isreplace=True)


    code = analyze_user_prompt(input_task, code, llm)

    
    code = remove_main(code)
    print('UPDATEDCODE')
    print(code)
    evaluate_code(code, message_history, markdown_df, llm)
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


def rewrite_user_prompt(input_task, llm):
    task = """
        Prompt:
        You are an advanced AI assistant.
        Rewrite the following prompt to make it concise, clear, and specific while retaining its original meaning.
        Provide only the rewritten version as your response—do not include explanations or additional commentary.

        User Prompt:
        {input_task}
            """ 
    prompt = PromptTemplate.from_template(task)
    chain = prompt | llm
    start_time = time.time()
    response = chain.invoke({"input_task": input_task})
    print("---Rewrite user response %s seconds ---" % (time.time() - start_time))
    print("Rrewrite Prompt Response")
    print(response)
    return response     


def analyze_user_prompt(input_task, code, llm):
    prompt_messages = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template("""
                    Here is some Python code:
                    {code}

                    And here is what the user wants the Python code to do:
                    {input_task}

                    Does the code do everything that the user requires?
                    """)])
    chain = prompt_messages | llm
    response = chain.invoke({"code": code, "input_task": input_task})
    print("DISCREPANCY ANALyzed")
    print(response)
    prompt_messages.append(AIMessage(content=response))
    compare_prompt = SystemMessagePromptTemplate.from_template("""
                    Change the code, based on your analysis, to meet all the user's requirements.                         
                    """)                  

    prompt_messages.append(compare_prompt)
    chain = prompt_messages | llm
    compare_response = chain.invoke({"code": code, "input_task": input_task})
    print("NEW CODE - Errors fixed")
    print(compare_response)
    code_prompt = SystemMessagePromptTemplate.from_template("""
                    What is the updated code? Or if there were, no changes to make, what was the original code?                         
                    """)
    prompt_messages.append(code_prompt)
    chain = prompt_messages | llm
    code_response = chain.invoke({"code": code, "input_task": input_task})
    print('Code Response--')
    print(code_response)
    code = extract_python_only(code_response)
    print('Update Code COde')
    print(code)
    return code



def rewrite_my_code_retrieve(code, change_request):
    system_prompt = (
        "Change the Python code based on my instructions. "
        "Use the data provided as context. "
        "Here is the Python code: {code} "
        "Here is the change request: {change_request} "
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
    result = rag_chain.invoke({"input": "Rewrite the code based on my change request", "code": code, "change_request": change_request})
    code = result['answer']
    return code


def rewrite_none_result(code, change_request):
    system_prompt = (
        "Change the Python code based on my instructions. "
        "Use the data provided as context. "
        "Here is the Python code: {code} "
        "Here is the change request: {change_request} "
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
    result = rag_chain.invoke({"input": "Rewrite the code based on my change request", "code": code, "change_request": change_request})
    code = result['answer']
    return code


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


def rerun_after_errorFF(code, error):
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


def rerun_after_error(code, error, message_history, markdown_df, llm):
    sys_message = [SystemMessagePromptTemplate.from_template(
        "The code is generating an error. Re-write the code so that it does not generate the error."
        "This is the code: {code}\n"
        "This is the error: {error}\n"
        "This is the data the code is written for: {dataset}"
    )]
    
    prompt = ChatPromptTemplate.from_messages(sys_message)
    #prompt = sys_message
    chain = prompt | llm
    response = chain.invoke({"code": code, "error": error, "dataset": markdown_df}) 
    print("RERUN _Response:")
    print(response)
    code = extract_python_only(response)
    print('codeonly -rerun')
    print(code)
    evaluate_code(code, message_history, markdown_df, llm)


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
def evaluate_code(code, message_history, markdown_df, llm):  # write methode to convert OBJ into non technical rrror to display to user
    try:                   
        exec(code, None, globals())
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
        rerun_after_error(code, exc_obj, message_history, markdown_df, llm)


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


def chunck_data(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    #chunks = DataFrameLoader(df,
    #                        page_content_column='VIN (1-10)').load()
    #data = df.to_string()
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


def new_or_old_F(client, user_input):
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


def check_true_or_false(s):
    s = s.lower()
    s_l = s.split()
    t = 0
    f = 0
    for w in s_l:
        if 'true' in w:
            t +=1
        if 'false' in w:
            f += 1
    print('t:', t)
    print('f:', f)
    if t > f:
        return True
    else:
        return False


def new_or_old(user_input, llm, retriever, message_history):
    new_or_old_sys_prompt = """
            Determine if the user wants to undo the changes that were made or continue with the current output.
            Output True if the user wants to undo the changes that were made and output False if the user wants to continue with the current output.
            Your response needs to be exactly: True or False
            \n\n
            """
    prompt_messages = [
        *message_history,
        HumanMessagePromptTemplate.from_template("{input}"),
        SystemMessagePromptTemplate.from_template(new_or_old_sys_prompt + "{context}")
    ]
    prompt = ChatPromptTemplate.from_messages(prompt_messages)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    results = rag_chain.invoke({"input": user_input})
    isRedo = results['answer']
    print("NEW OR OLD MODEL RESULT")
    print(isRedo)
    isRedo_clean = check_true_or_false(isRedo)
    print("CLEAN REDO")
    print(isRedo_clean)
    return isRedo_clean


def check_invoke_anal(response):
    words = response.split()
    for word in words:
        if 'guacamole' in word.lower():
            return True
    return False


def chatter(user_input, llm, retriever, chatter_history):
    chat_prompt = """
                    If you need to more context to answer the users question or if then say guacamole.

                """
    prompt_messages = [
        *chatter_history,        
        HumanMessagePromptTemplate.from_template("{input}"),
        SystemMessagePromptTemplate.from_template(chat_prompt)
    ]
    prompt = ChatPromptTemplate.from_messages(prompt_messages)
    #question_answer_chain = create_stuff_documents_chain(llm, prompt)
    #rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    #results = rag_chain.invoke({"input": user_input})
    #chatter_response = results['answer']
    chain = prompt | llm
    chatter_response = chain.invoke({"input": user_input})  
    print("CHATTER SAYS:")
    print(chatter_response)
    isAnal = check_invoke_anal(chatter_response)
    print("ANAL INVOKED?")
    print(isAnal)
    if isAnal:
        return 'True', chatter_history
    else:
        return chatter_response, chatter_history