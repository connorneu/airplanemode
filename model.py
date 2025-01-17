import os
import sys
import pandas as pd
import ollama
import traceback
import ast
import re
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
import copy



ui = None
llm_g = None
retriever = None
message_history_g = None
# CHANGE RPMPT BACK TO COMMENTEDF OUT THIS ONE IS NT WORKIONGGIT
SYS_INSTRUCTIONS = ("""You are a Python expert. 
            Write Python code to alter the dataset provided based on the user's statement.
            Be sure to check the column header names to match the spelling exactly.
            The dataset is located in a file named 'input_file.csv'.
            Follow these instructions carefully: 
            1. Read 'input_file.csv' into a pandas DataFrame named `data`.
            2. Create code that will generate a dataset which answers the users input statement with as much insight as possible.
            3. Save the final output DataFrame as 'doData_Output.csv'.
            4. Exclude any print statements from your response. The final output needs to includes all the calculations performed.   
            Remember: Keep the code simple. The user needs scripts that will execute correctly on the first try."""
        )
#("""You are a Python expert. 
#            Write Python code to do what the user asks.
#            Be sure to check the column header names to match the spelling exactly {column_headers}
#            Think carefully about the user's question. The user need your code to execute their instructions and generate the output as a csv file. 
#            The dataset is located in a file named 'input_file.csv'.
#            Follow these instructions carefully: 
#            1. Read 'input_file.csv' into a pandas DataFrame named `data`.
#            2. Create code that will generate a dataset which answers the users input statement with as much insight as possible.
#            3. Save the final output DataFrame as 'doData_Output.csv'.
#            Remember: Keep the code simple. The user needs scripts that will execute correctly on the first try."""
#        )


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


def reset_prompt_history():
    message_history = ChatPromptTemplate.from_messages([SystemMessage(content=SYS_INSTRUCTIONS)])
    message_history.append(SystemMessagePromptTemplate.from_template("This is the DataFrame the user is analyzing: {dataset}"))
    message_history.append(HumanMessagePromptTemplate.from_template("{input_task}"))
    return message_history


def update_prompt_with_history(message_history):
    #prompt_messages = [
    #    *message_history,
    #    SystemMessagePromptTemplate.from_template("This is the DataFrame the user is analyzing: {dataset}"),
    #    HumanMessagePromptTemplate.from_template("{input_task}"),
    #]
    message_history.append(SystemMessagePromptTemplate.from_template("This is the DataFrame the user is analyzing: {dataset}"))
    message_history.append(HumanMessagePromptTemplate.from_template("{input_task}"))
    #return ChatPromptTemplate.from_messages(prompt_messages)
    return message_history


def chatter(user_input, llm, ui):
    #ChatPromptTemplate.from_messages([SystemMessage(content=self.system_prompt)])
    chatter_history = ChatPromptTemplate.from_messages([SystemMessage(content="""You are a helpful assistant who helps user better understand their data.
                                              You are part of a team of two assistants. Your have a broad knowledge of a lot of things and are good at answering
                                                users questions. If a user asks a question that will require an analysis of the uploaded data then say 'bananhamock'.
                                                Otherwise, answer the user and be helpfull and assure them that you are very capable of analyzing their data.""")])
    chatter_history.append((HumanMessagePromptTemplate.from_template("{user_input}")))
    chain = chatter_history | llm
    response = chain.invoke({"user_input": user_input})
    chatter_history.append(response)
    return response
    #async for chunk in chain.astream({"user_input": user_input}):
    #    print(chunk, end="|", flush=True)
    #    ui.ai_response(chunk, end="|", flush=True)
    #from langchain_core.output_parsers import StrOutputParser


def calculate_history_length(history, markdown_df, input_task, column_names):
    prompt_as_string = history.format(
    dataset=markdown_df,
    input_task=input_task,
    column_names=column_names,
    )
    print("Prompt Length:", len(prompt_as_string), "(" + str(len(prompt_as_string)/1.5) + " tokens)")
    print(prompt_as_string)

def change_prompt_to_affirmative(prompt, llm):
    print("inouttas")
    print(prompt)
    prompt_template = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template("""Rephrase the question so that it is a query.
Question: {prompt}""")])
    print("REALTALK")
    s = prompt_template.format(prompt=prompt)
    print(s)
    chain = prompt_template | llm
    response = chain.invoke({"prompt": prompt})
    print("Changed Prompt:")
    print(response)
    return response


def run_model(user_input, llm, message_history, column_names, markdown_df, ui_g, rerun, eval_attempts = 0):

    #import langchain
    #langchain.debug = True 

    print("MARKDOWNDF")
    print(markdown_df)

    global ui
    ui = ui_g
    input_path = user_input['import_file']
    input_task = user_input['user_input']
    output_path = user_input['output_path']
    message_history = update_prompt_with_history(message_history)   
    solved = False
    print("Starting message history")
    print(message_history)
    print('Input path:', input_path)
    print('Output path:', output_path)
    input_path = escape_filepath(input_path)
    output_path = escape_filepath(output_path)
    print('Input path escaped:', input_path)
    print('Output path escaped:', output_path)
    #input_task = change_prompt_to_affirmative(input_task, llm)

    while not solved and eval_attempts < 10:
        print("starting")
        timetotal = time.time()
        print("isSolved:", solved)
        firststart = time.time()
        calculate_history_length(message_history, markdown_df, input_task, column_names)
        chain = message_history | llm
        if eval_attempts > 0:
            response = chain.invoke({"dataset": markdown_df, "input_task": input_task, "code": code})
        else:     
            response = chain.invoke({"dataset": markdown_df, "input_task": input_task}) 
        print("Response:")
        print(response)
        message_history.append(AIMessage(content=response))
        first_response_time = time.time() - firststart
        print("---First Response Time %s seconds ---" % (time.time() - firststart))
        code = extract_python_only(response)
        code = update_paths(code, input_path, output_path)
        #code = remove_elem(code, '#')
        code = remove_elem(code, 'input(', isreplace=True)
        timeanal = time.time()
        #if False:
        #if first_response_time < 10:
        #code, message_history = analyze_user_prompt(input_task, code, llm, column_names, message_history, markdown_df, input_path, output_path)
        print("---Anal Time %s seconds ---" % (time.time() - timeanal))
        #code = find_print_line_commas(code)
        #code = replace_prints(code)
        code = remove_main(code)
        timeeval = time.time()
        eval_attempts, message_history = evaluate_code(code, message_history, markdown_df, llm, input_task, eval_attempts, input_path, output_path)
        print("---Eval Time %s seconds ---" % (time.time() - timeeval))
        print("RUN MODEL EVAL:", eval_attempts)
        #if False:
        if not check_output_exists(output_path):
            print('No result file exists.')
            eval_attempts += 1
            print("No file evals:", eval_attempts)
            message_history = reset_prompt_history()
            print('messaghistery')
            print(message_history)
        else:
            print('SOLVED.')
            solved = True
        print("---TOTAL TIME %s seconds ---" % (time.time() - timetotal))
    if not solved:
        print("Yo Filed.")
        explanation = 'I was unable to process your request. Please rephrase your question and try again.'        
    else:
        print('run mode complete.')
        explanation = 'Here\'s your data so far.'
    return message_history, code, markdown_df, explanation


def check_output_exists(outpath):
    if os.path.isfile(outpath):
        return True
    else:
        return False


def no_file_generated(llm, message_history, markdown_df, code, input_task, output_path):
    prompt = SystemMessagePromptTemplate.from_template("""
    The code that you wrote must output the result as a DataFrame here: {output_path}
                                                       \n
    Please change the code so that the result that it generates is saved as a csv.
                                                           """)
    message_history.append(prompt)
    chain = message_history | llm
    response = chain.invoke({"dataset": markdown_df, "code": code, "input_task": input_task, "output_path": output_path})
    print("NO File Generated Response")
    print(response)
    code = extract_python_only(response)
    return code


def remove_elem(code, elem, isreplace=False):
    clean = ''
    for line in code.split('\n'):
        if not elem in line:
            clean += line + '\n'
        else:
            if isreplace:
                clean += 'pass' + '\n'
    return clean


def analyze_user_prompt(input_task, code, llm, column_names, message_history, markdown_df, input_path, output_path):
    timehowis = time.time()
    anal_prompt = SystemMessagePromptTemplate.from_template("""
                    Here is some Python code:
                    {code}

                    And here is what the user wants the Python code to do:
                    {input_task}

                    Does the code do everything that the user requires?
                    Please note that I'm not asking about robustness.
                    There's no need to create any form of validation.
                    Simply review if what the user asked is accomplished by the code without trying to unecessarily improve the code. 
                    """)
    message_history.append(anal_prompt)
    chain = message_history | llm
    response = chain.invoke({"dataset": markdown_df, "code": code, "input_task": input_task, "column_headers": column_names})
    print("DISCREPANCY ANALyzed")
    print(response)
    print("--- How Is %s seconds ---" % (time.time() - timehowis))
    message_history.append(AIMessage(content=response))
    timerevise = time.time()
    compare_prompt = SystemMessagePromptTemplate.from_template("""
                    Change the code, based on your analysis, to meet all the user's requirements.
                    """)                  
    message_history.append(compare_prompt)
    chain = message_history | llm
    compare_response = chain.invoke({"dataset": markdown_df, "code": code, "input_task": input_task, "column_headers": column_names})
    print("NEW CODE - Errors fixed")
    print(compare_response)
    print("--- Make change %s seconds ---" % (time.time() - timerevise))
    message_history.append(AIMessage(content=compare_response))
    code = extract_python_only(compare_response)
    return code, message_history


def add_back_output_pathF(code, output_path):
    print('missing output path.')
    code_clean = ''
    lines = code.split('\n')
    for line in lines:
        if '.to_csv(' in line:
            print('Qline:', line)
            print('QQpa:', output_path)
            input_line = replace_string_between_quotes(line, output_path)
            print('output line:', input_line)
            code_clean += input_line + '\n'
            print('output line added back.')
        else:
            code_clean += line + '\n'          
    return code_clean


def add_back_output_path(code, output_path):
    code_clean = ''
    lines = code.split('\n')
    for line in lines:
        if '.to_csv(' in line:
            #if output_path not in line:
            print("missing oot path")
            print('line:', line)
            print('pa:', output_path)
            output_line = replace_string_between_quotes(line, output_path)
            print('oot line:', output_path)
            code_clean += output_line + '\n'
            print('output line added back.')
        else:
            code_clean += line + '\n'
    return code_clean


def add_back_input_pathF(code, input_path):
    print('missing input path.')
    code_clean = ''
    lines = code.split('\n')
    for line in lines:
        if '.read_csv(' in line:
            print('line:', line)
            print('pa:', input_path)
            input_line = replace_string_between_quotes(line, input_path)
            print('input line:', input_line)
            code_clean += input_line + '\n'
            print('input line added back.')
        else:
            code_clean += line + '\n'       
    return code_clean


def add_back_input_path(code, input_path):
    code_clean = ''
    lines = code.split('\n')
    for line in lines:
        if '.read_csv(' in line:
            #if input_path not in line:
            print("missing input path")
            print('line:', line)
            print('pa:', input_path)
            input_line = replace_string_between_quotes(line, input_path)
            print('input line:', input_line)
            code_clean += input_line + '\n'
            print('input line added back.')
        else:
            code_clean += line + '\n'
    return code_clean


def replace_string_between_quotes(text, replacement):
    # Regular expression to find text between single or double quotes
    pattern = r'([\'"])(.*?)(\1)'
    m = re.match(pattern, text, re.M)
    return re.sub(pattern, r'\1' + replacement + r'\1', text, count=1)


def is_python(line):
   try:
       if len(line.strip())>0:
           ast.parse(line)
           return True
       else:
           return False
   except SyntaxError:
       return False


def escape_filepath(filepath):
    if '\\' in filepath:
        filepath = filepath.replace('\\', '/')
    return filepath


def escape_generated_filepaths(code):
    code_cleaned = ''
    lines = code.split('\n')
    for line in lines:
        if '.read_csv(' in line or '.to_csv(' in line:
            if '\\' in line:
                line = line.replace('\\', '/')
        code_cleaned += line + '\n'
    return code_cleaned


def extract_python_only(code):
    code = remove_triple_quote_comments(code)
    code = escape_generated_filepaths(code)
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


def remove_triple_quote_comments(code):
    num_trip_quotes = code.count('"""')
    if num_trip_quotes > 1:
        code_cleaned = ''
        lines = code.split('\n')
        in_trip_quote = False
        i = 0
        while i < len(lines):
            if '"""' in lines[i]:
                if lines[i].strip().startswith('"""') and lines[i].strip().endswith('"""'):
                    pass
                elif not in_trip_quote:
                    in_trip_quote = True
                else:
                    in_trip_quote = False
                    i += 1
            if not in_trip_quote:
                code_cleaned += lines[i] + '\n'
            i += 1
        return code_cleaned
    else:
        return code


def update_paths(code, input_path, output_path):
    code = code.replace('doData_Output.csv', output_path)
    if 'input_file.csv' in code:
        code = code.replace('input_file.csv', input_path)
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


def rerun_after_error(code, error, message_history, markdown_df, llm, input_task, eval_attempts, input_path, output_path):
    error_message = SystemMessagePromptTemplate.from_template(
        "This is Python code:"
        "{code}\n"
        "This Python code is generating this error: {error}\n"
        "Why is this python code generating this error?"
    )
    print("Zthe ERROR: ", error)
    message_history.append(error_message)
    chain = message_history | llm
    error_response = chain.invoke({"input_task": input_task, "code": code, "error": error, "dataset": markdown_df}) 
    print("RERUN _Response:")
    print(error_response)

    resolve_message = SystemMessagePromptTemplate.from_template(
        "Change the code, based on your analysis, to resolve the error that the code is generating."
    )
    message_history.append(resolve_message)
    chain = message_history | llm
    resolve_response = chain.invoke({"input_task": input_task, "code": code, "error": error, "dataset": markdown_df}) 
    print('R3solve error response')
    print(resolve_response)
    code = extract_python_only(resolve_response)
    print('codeonly -rerun')
    print(code)
    evaluate_code(code, message_history, markdown_df, llm, input_task, eval_attempts, input_path, output_path)


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
                #stripline = lines[i].strip()
                # no exception handling if try catch in main
                s = lines[i].replace("\t", "    ")
                num_leading_whites = len(s) - len(s.lstrip(' '))
                print('LINE', s)
                print(num_leading_whites)
                if num_leading_whites > 0 and num_leading_whites % 4 == 0:
                    print("Q", s)
                    s = s[4:]
                    print("P", s)
                clean += s + '\n'
            i += 1
        return clean
    else:
        print('no main statement to clean')
        return code


# exec problems https://stackoverflow.com/questions/4484872/why-doesnt-exec-work-in-a-function-with-a-subfunction
def evaluate_code(code, message_history, markdown_df, llm, input_task, eval_attempts, input_path, output_path):  # write methode to convert OBJ into non technical rrror to display to user
    global ui
    print("INPOUT", input_path)
    print("OUTPOOT", output_path)
    try:         
        #if not input_path in code:
        code = add_back_input_path(code, input_path)
        #if not output_path in code:
        code = add_back_output_path(code, output_path)    
        print('final final code')
        print(code)      
        exec(code, None, globals())
        print('Code execution complete.')
        return eval_attempts, message_history
    except Exception as e:
        print("CODE FAILURE")
        eval_attempts += 1
        print('eval attempt:', eval_attempts)        
        trace = traceback.format_exc()
        print("trace -- ")
        print(trace)
        print('exc infor')
        print(sys.exc_info())
        print('end exc')
        error_message = 'The code generated this error: \n' + str(trace)
        print("ERROR MESSAGE:")
        print(error_message)
        error_prompt = HumanMessagePromptTemplate.from_template(error_message)
        message_history.append(error_prompt)
        return eval_attempts, message_history
        #if eval_attempts > 2:
        #    print("Failed 3 times. Restarting Process.")
        #    message_history = [SystemMessage(content=SYS_INSTRUCTIONS)]
        #    run_model(input_task, llm, message_history, markdown_df, eval_attempts = -1)
        #else:
        #    rerun_after_error(code, exc_obj, message_history, markdown_df, llm, input_task, eval_attempts, input_path, output_path)


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
    llm = OllamaLLM(model="llama3.2", temperature=1) # cas/llama-3.2-3b-instruct
    return llm


def build_llm_cpp():
    from langchain_community.llms import LlamaCpp
    from llama_cpp import Llama
    llm = Llama.from_pretrained(
    repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
    filename="*Llama-3.2-3B-Instruct-IQ3_M.gguf",
    verbose=False
    )
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


# NOTES
# add validation model to ensure that no files are accessed outside /work directory