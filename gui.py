from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QTextBrowser, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel, QScrollArea, QFileDialog, QTableWidget, QTableWidgetItem, QSizePolicy
from PyQt6.QtCore import Qt, QTimer, QSize, QTextStream, QFile, QUrl
from PyQt6.QtGui import QIcon, QPixmap, QFontMetrics, QMovie
import pandas as pd
import sys
import model
import os
from dateutil.parser import parse
import traceback
import ast
import csv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
import datetime
import time
# from pandas.tseries.api import guess_datetime_format
import numpy as np
from random import randrange
import asyncio
import time
import multiprocessing


SCREENWIDTH = 0.5
SCREENHEIGHT = 0.7

class ChatInterface(QMainWindow):
    def __init__(self):
        super().__init__()


        self.setWindowTitle("Airplane Mode AI")
        self.screen = QApplication.primaryScreen().size()
        self.setGeometry(100, 100, int(self.screen.width() * SCREENWIDTH), int(self.screen.height() * SCREENHEIGHT))
        self.setStyleSheet("background-color: #2b2b2b;")

        #self.setGeometry(300, 100, 600, 500)

        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout for all chat messages
        self.chat_layout = QVBoxLayout()
        self.chat_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        ai_intro = """
Welcome to AirplaneMode AI\n
A tool to analyze your data privately using AI.\n
Upload your data to begin.
"""
        self.ai_response(ai_intro)
        #self.psudo_type(ai_intro)
        
        # Scroll area to hold chat layout
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        chat_container = QWidget()
        chat_container.setLayout(self.chat_layout)
        self.scroll_area.setWidget(chat_container)
        self.scroll_area.setStyleSheet("background-color: #444444;")
        
        # File drop area setup
        self.file_drop_area = QLabel("Click here or drag and drop a file to start", self)
        self.file_drop_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_drop_area.setStyleSheet("border: 2px dashed #888; padding: 20px; font: 20px 'Ubuntu'; color: #888; background-color: #2b2b2b; color:white;")
        self.file_drop_area.setAcceptDrops(True)
        
        # Input box and submit button setup (initially hidden)
        self.input_box = QLineEdit()
        self.input_box.setStyleSheet("color: white; font: 14px 'Ubuntu';")
        self.input_box.setPlaceholderText("Ask a question...")
        font = self.input_box.font()  
        font.setPointSize(14) 
        font_metrics = QFontMetrics(font)
        line_height = font_metrics.lineSpacing()
        desired_height = 2 * line_height  # Height for two lines of text
        self.input_box.setFixedHeight(35)

        self.input_box.returnPressed.connect(self.handle_submit)
        
        self.input_box.setVisible(False)
        
        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.handle_submit)
        self.submit_button.setVisible(False)
        self.submit_button.setStyleSheet("color: white; font: 14px 'Ubuntu';")
        self.submit_button.setFixedHeight(28)
        
        # Additional file upload icon
        self.upload_icon = QPushButton()
        self.upload_icon.setIcon(QIcon("upload_icon.png"))  # Replace with path to an upload icon image
        self.upload_icon.setIconSize(QPixmap("upload_icon.png").size())
        self.upload_icon.clicked.connect(self.upload_additional_file)
        self.upload_icon.setVisible(False)
        self.upload_icon.setText('Upload New File')
        self.upload_icon.setStyleSheet("color: white; font: 14px 'Ubuntu';")
        self.upload_icon.setFixedHeight(28)

        # Bottom layout for input, submit button, and upload icon
        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.input_box)
        bottom_layout.addWidget(self.submit_button)
        bottom_layout.addWidget(self.upload_icon)
        
        # Main layout that combines chat display, file drop area, and input area
        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.scroll_area)
        self.main_layout.addWidget(self.file_drop_area)
        self.main_layout.addLayout(bottom_layout)
        
        
        # Set main layout to central widget
        central_widget.setLayout(self.main_layout)
        #"Write the result of the Python code to a DataFrame and export it as a csv called doData_Output.csv. "

        self.system_prompt = ("""You are a Python expert. 
            The user needs your help analyzing their data.
            Write Python code to help the user better understand their data.
            Write simple code to change the dataset based on your best understanding of the users request.
            Be sure to check the column header names to match the spelling exactly.
            The dataset is located in a file named 'input_file.csv'.
            Follow these instructions carefully: 
            1. Read 'input_file.csv' into a pandas DataFrame named `data`.
            3. Save the final output DataFrame as 'doData_Output.csv'.
            4. Exclude any data validations and exception handling. Write simple code to generate a solution for the user.
            5. Exclude print statements as the user needs their output as a DataFrame.
            6. Force any kind of data type validation.             
            """
        )

        self.data1 = None
        self.data1_col = None
        self.data1_trunc = None
        self.data1_result = None
        self.data1_filepath = None
        self.client = model.build_client()
        self.message_history = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(self.system_prompt)])
        self.docsplits = None
        self.embeddings = None
        self.llm = None
        self.retriever = None
        self.work_dir = None
        self.markdown_df = None
        self.rerun = False
        self.old_message_history = None
        self.old_df_markdown = None

        # create directory for working files
        if not os.path.exists('work'):
            os.makedirs('work')
        self.work_dir = os.path.join(os.getcwd(), 'work')



        if False:
            self.movie_label = QLabel(self)
            load_gif_path = resource_path(os.path.join('images', 'load.gif'))
            self.movie = QMovie(load_gif_path)
            self.movie_label.setGeometry(400, 400, 250, 250)
            self.movie.setScaledSize(self.movie_label.size()) 
            self.movie_label.setMovie(self.movie)
            self.chat_layout.addWidget(self.movie_label)
            self.movie.start()
        



    def psudo_type(self, text):
        self.text = text
        print(self.text)
        self.index = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_text)
        self.timer.start(30)
        isLabel = False
        self.label = QLabel("")
        self.label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.label.setStyleSheet("color: #bea5f0; background-color: #444444; padding: 5px; border-radius: 25px;font: 16px 'Ubuntu';")
        self.label.setWordWrap(True)
        size_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.label.setSizePolicy(size_policy)
        self.label.setMaximumWidth(int((int(self.screen.width() * SCREENWIDTH)) * .45))
        self.label.setFixedHeight(self.label.sizeHint().height())
        self.chat_layout.addStretch()  # add some space at the top
        self.chat_layout.addWidget(self.label)
        #self.chat_layout.addStretch()
        


    # update text for psudo typing
    def update_text(self):
        if not self.text:
            return
        current_char = self.text[self.index]
        self.label.setText(self.label.text() + current_char)
        self.label.setFixedHeight(self.label.sizeHint().height())
        self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())

        # increment index
        self.index += 1

        # stop after text has been fully displayed
        if self.index >= len(self.text):
            self.timer.stop()


    def mousePressEvent(self, event):
        # Detect if the file drop area is clicked
        if self.file_drop_area.underMouse():
            self.open_file_dialog()            


    def dragEnterEvent(self, event):
        # Accept drag event if it contains files
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        # Handle file drop
        files = [url.toLocalFile() for url in event.mimeData().urls()]
        if files:
            self.handle_file_upload(files[0])


    def open_file_dialog(self):
        # Open file dialog for selecting a file
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
            self.handle_file_upload(file_path)


    def read_file(self, filepath):
        if '.xl' in filepath:
            try:
                return pd.read_excel(filepath) 
            except Exception as e:
                return pd.read_excel(filepath, encoding='windows-1252')
        elif '.csv' in filepath:
            try:
                return pd.read_csv(filepath)
            except:
                return pd.read_csv(filepath, encoding='windows-1252')
        elif '.txt' in filepath:
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(filepath)
            print('dialect', dialect)
            return pd.read_csv(filepath, sep=dialect) #untested
        else:
            print('Unsuported file type')
            return 'Unsuported file type'
        

    def create_tiny_series(self, df, col):
        df = pd.Series(df[col].unique()).head()
        return df


    def date_format_guesser(self, df_tiny):
        for val in df_tiny.values.tolist():
            guessed_format = guess_datetime_format(str(val))
            if guessed_format is not None:
                return guessed_format
        return None


    def update_date_column_types(self, df, col):
        #try:
        df_tiny = self.create_tiny_series(df, col)
        try:
            df[col+'_t'] = df_tiny.apply(self.is_date)
        except:
            pass
        try:
            if df[col+'_t'].any():
                guessed_format = None # ~!
                #guessed_format = self.date_format_guesser(df_tiny) # ! removed because downgraded to pandas version equal to llama3.2 date
                if guessed_format is not None:
                    try:
                        df[col] = pd.to_datetime(df[col], format=guessed_format)
                    except:
                        print('Failed to convert to datetime')
                else:
                    print("unable to guess format")
                    #df[col] = pd.to_datetime(df[col], errors='ignore')
                #df.drop(col+'_t', axis=1, inplace=True)
            df.drop(col+'_t', axis=1, inplace=True)        
        except:
            pass
        return df


    def is_date(self, val):
        #param fuzzy: bool, ignore unknown tokens in string if True
        try: 
            parse(str(val), fuzzy=False)
            return True

        except ValueError:
            return False


    def num_len(self, val):
        if len(str(val)) > 0:
            return True
        else:
            return False


    def shorten_num_cols(self, df):
        num_cols = df.select_dtypes(include=['int64', 'float64'])
        for num_col in num_cols:
            p = df[num_col].apply(self.num_len).any()
            if p:
                df[num_col] = 1
        return df    
    

    def calc_tokens(self, data):
        # to string truncates values > 50 
        # changed from to string to markdown
        x = int(len(data.to_markdown()) / 1.5) #llama3.2 (ollama) says hugging face says 1.5
        return x
    

    def replace_col_vals_unique(self, df):
        num_rows = df.shape[0]
        for col in df.columns:
            if df[col].dtype == object:
                ul = df[col].unique().tolist()
                if len(ul) < num_rows:
                    while len(ul) < num_rows:
                        ul.append(ul[randrange(len(ul))])
                else:
                    ul = ul[:num_rows]
                df[col] = ul
        return df


    # 1 token ~=  4 chars
    # https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
    # llama3.2:1b says that each word is 2 characters
    def trunc_data(self, data, max_col, max_col_u):
        print("MAX COL", max_col, max_col_u)
        if max_col is not None:
            df = self.replace_col_vals_unique(data)
            df = df.head(data[max_col].nunique(dropna=False)) # changed this from head(max_col_U because depending on dataset it is one off for unknown reason            
            df[max_col] = data[max_col].unique()            
        else:
            df = data
        my_context_size = 2000
        # llama3.2:1b = context limit is 500 - 750 words
        # system prompts are ~800 characters
        # llama3.2:1b words are 2 chars long3
        # 800 characters is 400 words of 750 limit
        # my_context_size needs to be 300
        #my_context_size = 500
        orig_len = self.calc_tokens(df)
        print('originallength')
        print(orig_len)
        if orig_len < my_context_size:
            print("less than 3000")
            return df
        else:
            cut_x = 10000
            curr_len = orig_len
            while curr_len > my_context_size:
                if cut_x < 1:
                    print("cut forced to 1")
                    return df.head(1)
                print("Curr", curr_len, cut_x)
                df = df.head(cut_x)
                curr_len = self.calc_tokens(df)
                if curr_len < my_context_size:
                    print('returning', curr_len)
                    print('numrows:',cut_x)
                    print('numtokens', int(len(df.to_string()) / 4))
                    print('num characters', int(len(df.to_string())))
                    return df
                else:
                    if cut_x > 1000:
                        cut_x -= 1000
                    elif cut_x > 101:
                        cut_x -= 100
                    elif cut_x > 11:
                        cut_x -= 10
                    else:
                        cut_x -= 1


    def minimize_embedded_df(self, df):
        max_col_u = 0
        max_col = None
        df = df.head(1000)
        # if type is not str or num check if its date
        print(df.dtypes)
        for col in df.columns:
            if df[col].dtype == object:
                df = self.update_date_column_types(df, col)
                if df[col].dtype == object:
                    cu = df[col].nunique()
                    #print('UNIQUENESS', col, cu)
                    #print('num vals', df.shape[0])
                    #print('cuni', len(df[col].values.tolist()))
                    #print('pcnt unqieuns', (cu / df.shape[0]))
                    if cu > max_col_u and (cu / df.shape[0]) < 0.95:
                        max_col_u = cu
                        max_col = col
        df = self.shorten_num_cols(df)
        df = self.trunc_data(df, max_col, max_col_u)
        return df

    def handle_file_upload(self, file_path):
            self.file_drop_area.setVisible(False)
            self.input_box.setVisible(True)
            self.submit_button.setVisible(True)
            self.upload_icon.setVisible(True)
            
            file_label = QLabel(f"File uploaded: {file_path}")
            file_label.setStyleSheet("color: #ffffff; background-color: #555555; padding: 10px; border-radius: 5px;")
            file_label.setFixedHeight(file_label.sizeHint().height())
            file_label.setFixedWidth(file_label.sizeHint().width())
            self.chat_layout.addWidget(file_label, alignment= Qt.AlignmentFlag.AlignRight)
            try:
                timeread = time.time()
                data = self.read_file(file_path)
                self.data1 = data.copy()
                path, filename = os.path.split(file_path)
                self.data1_filepath = os.path.join(self.work_dir, filename)
                self.data1.to_csv(self.data1_filepath, index=False)
                self.data1_trunc = self.minimize_embedded_df(self.data1)
                self.data1_col = list(self.data1.columns)    
                #self.llm = model.build_llm()
                self.llm = model.build_llm_cpp()
                self.markdown_df = self.data1_trunc.to_markdown()
                print(self.markdown_df)
                if not check_machine_competence():
                    print("No RAG")
                    isSuggestion_success = False
                    trycount = 3
                else:
                    suggestions, self.docsplits, self.embeddings, self.llm, self.retriever, self.markdown_df = model.suggest_actions(self.data1_trunc)
                    trycount = 0
                    isSuggestion_success = False
                while trycount < 3:
                    try:
                        suggestions = ast.literal_eval(suggestions)
                        suggestions = [n.strip() for n in suggestions]
                        isSuggestion_success = True
                        break
                    except:
                        print('SUggestion FAILED', trycount)
                        print(suggestions)
                        trycount += 1
                if not isSuggestion_success:
                    suggestions = ['Calculate the difference between dates', 
                                'Filter your dataset to based on complicated requirements',
                                'Discover relationships between different columns']
                self.display_data_preview(data)
                self.display_suggestion_buttons(suggestions)
                self.psudo_type("Nice Data! Here are some suggestions of what kind of analysis you can do.")
                print("--- Read Time %s seconds ---" % (time.time() - timeread))
            except Exception as e:
                traceback.print_exc()
                # Display an error message if the file could not be read
                error_label = QLabel(f"Could not read file: {e}")
                error_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
                error_label.setStyleSheet("color: #ff0000; padding: 10px; border-radius: 5px;")
                self.chat_layout.addWidget(error_label)


    def newline_suggestion(self, suggestion):
        ns = ''
        char_count = 0
        for c in suggestion:
            if char_count > 100 and c == ' ':
                ns += '\n'
                char_count = 0
            ns += c
            char_count += 1
        return ns
            

    def display_suggestion_buttons(self, suggestions):
        button_layout = QVBoxLayout()
        for suggestion in suggestions:
            suggestion = self.newline_suggestion(suggestion)
            button = QPushButton(suggestion)
            button.setStyleSheet("background-color: #2b2b2b; color: #ffffff; padding: 10px; border-radius: 5px;")
            size_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
            button.setSizePolicy(size_policy)
            button.setMaximumWidth(int((int(self.screen.width() * SCREENWIDTH))))
            button.setFixedHeight(100)
            button.clicked.connect(lambda _, s=suggestion: self.handle_suggestion_click(s))
            button_layout.addWidget(button)

        button = QPushButton('More suggestions...')
        size_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        button.setSizePolicy(size_policy)
        button.setMaximumWidth(int((int(self.screen.width() * SCREENWIDTH))))
        button.setFixedHeight(40)
        button.clicked.connect(lambda _, s=suggestion: self.handle_suggestion_click(s))
        button_layout.addWidget(button)

        button.setStyleSheet("background-color: #2b2b2b; color: #ffffff; padding: 10px; border-radius: 5px;")

        button_container = QWidget()
        button_container.setLayout(button_layout)
        self.chat_layout.addWidget(button_container, alignment=Qt.AlignmentFlag.AlignLeft)


    def handle_suggestion_click(self, suggestion):
        # Display the selected suggestion as a user input
        user_label = QLabel(f"User selected: {suggestion}")
        print("USER SELECTED SUGGESTION:", suggestion)
        user_label.setStyleSheet(
            "color: #00ffcc; background-color: #333333; padding: 5px; border-radius: 5px; font: 16px 'Ubuntu';"
        )
        user_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.chat_layout.addWidget(user_label)

        # Trigger the model based on the suggestion
        import_path = self.data1_filepath
        output_path = os.path.join(self.work_dir, 'doData_Output.csv')
        if os.path.exists(output_path):
            import_path = output_path
            print('doData file exists. Import path equals doData_Output.csv')
        model_input = {"import_file":import_path, "user_input": suggestion, "output_path": output_path}
        data_columns = list(self.data1_trunc.columns)
        print("COLUMN NAMES")
        print(data_columns)
        self.message_history, code, self.markdown_df, explanation = model.run_model(model_input, self.llm, self.message_history, data_columns, self.markdown_df, self, self.rerun)
        self.handle_response(output_path, code, explanation)


    def display_data_preview(self, data):
        # Limit data preview to first 10 rows
        preview_data = data.head(50)
        
        # Create a QTableWidget for displaying the data preview
        table = QTableWidget(preview_data.shape[0], preview_data.shape[1])
        table.setHorizontalHeaderLabels(preview_data.columns)
        table.setVerticalHeaderLabels([str(i) for i in preview_data.index])

        # Set table width to 80% of the window width and align to the left
        table.setFixedWidth(int(self.screen.width() * SCREENWIDTH) - 300)
        table.setFixedHeight(250)
        table.setStyleSheet("background-color: #444444; color: #ffffff; gridline-color: #555555;")

        # Populate the table with data
        for row in range(preview_data.shape[0]):
            for col in range(preview_data.shape[1]):
                item = QTableWidgetItem(str(preview_data.iat[row, col]))
                table.setItem(row, col, item)

        # Adjust table policies for scrolling and size
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Align the table to the right in the chat layout
        table_container = QHBoxLayout()
        table_container.addWidget(table)
        table_container.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.chat_layout.addLayout(table_container)
    

    def reset_result(self):
        reset_msg = "Oh! Sorry about that. I'm still learning. I've reset your data to your original uploaded data. Please decribe what you'd like to do."
        self.markdown_df = self.old_df_markdown
        self.message_history = self.old_message_history
        import_path = self.data1_filepath
        print('reset import path:', import_path)
        try:
            os.remove(os.path.join(self.work_dir, 'doData_Output.csv'))
        except:
            pass
        data = self.read_file(import_path)
        self.display_data_preview(data)
        self.psudo_type(reset_msg)


    def display_result_data(self, data):
        # Limit data preview to first 10 rows
        preview_data = data.head(50)
        
        # Create a QTableWidget for displaying the data preview
        table = QTableWidget(preview_data.shape[0], preview_data.shape[1])
        table.setHorizontalHeaderLabels(preview_data.columns)
        table.setVerticalHeaderLabels([str(i) for i in preview_data.index])

        # Set table width to 80% of the window width and align to the left
        table.setFixedWidth(int(self.screen.width() * SCREENWIDTH) - 300)
        table.setFixedHeight(250)
        table.setStyleSheet("background-color: #444444; color: #ffffff; gridline-color: #555555;")

        # Populate the table with data
        for row in range(preview_data.shape[0]):
            for col in range(preview_data.shape[1]):
                item = QTableWidgetItem(str(preview_data.iat[row, col]))
                table.setItem(row, col, item)

        # Adjust table policies for scrolling and size
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)


        # Reset
        download_button = QPushButton()
        download_button.setStyleSheet('QPushButton {background-color: #D3D3D3; color: black;}')
        download_button.setText('Not what you wanted? Press to RESET')
        download_button.setIcon(QIcon("download_icon.png"))  # Replace with the path to your download icon
        download_button.setToolTip("Reset")
        download_button.clicked.connect(self.reset_result)


        # Layout for the table and download button
        table_layout = QVBoxLayout()
        table_layout.addWidget(table)
        table_layout.addWidget(download_button, alignment=Qt.AlignmentFlag.AlignRight)


        # Align the table to the left in the chat layout
        table_container = QHBoxLayout()
        table_container.addLayout(table_layout)
        table_container.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.chat_layout.addLayout(table_container)
    

    def download_table(self, data):
        try:
            # Get the path to the user's Downloads folder
            downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
            file_name = "exported_table.csv"  # Default file name

            # Save the table data to the CSV file in the Downloads folder
            data.to_csv(file_name, index=False)
            self.ai_response(f"Table successfully saved to {file_name}!")
        except Exception as e:
            self.ai_response(f"Error saving table: {e}")


    def upload_additional_file(self):
        # Open file dialog for additional file upload
        self.open_file_dialog()


    def handle_submit(self):
        try:
            user_input = self.input_box.text()
            if user_input.strip():
                user_label = QLabel()
                user_label.setStyleSheet(
                    "color: #00ffcc; background-color: #333333; padding: 5px; border-radius: 5px; font: 16px 'Ubuntu';"
                )
                user_label.setWordWrap(True)
                user_label.setText(f"User: {user_input}")
                size_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
                user_label.setSizePolicy(size_policy)
                user_label.setMaximumWidth(int((int(self.screen.width() * SCREENWIDTH)) * .8))
                user_label.setFixedHeight(user_label.sizeHint().height())
                self.chat_layout.addWidget(user_label, alignment= Qt.AlignmentFlag.AlignRight)
                self.input_box.clear()

                print('user says:', user_input)
                import_path = self.data1_filepath
                output_path = os.path.join(self.work_dir, 'doData_Output.csv')
                if os.path.exists(output_path):
                    print("Howtput exists")
                    # rename doDataFile to something else so when run_model checks if output exists it wont be false positive
                    import_path_rerun_filename = os.path.join(self.work_dir, 'doData_Output_rerun.csv')
                    os.rename(output_path, import_path_rerun_filename)
                    import_path = import_path_rerun_filename
                print('Inputpath:', import_path)
                print('Outputpath:', output_path)
                model_input = {"import_file":import_path, "user_input":user_input, "output_path": output_path}
                self.data1_trunc.to_csv(os.path.join(self.work_dir, 't.csv'))
                data_columns = list(self.data1_trunc.columns)
                #import asyncio
                #asyncio.run(model.chatter(user_input, self.llm, self))
                #chatter_response = model.chatter(user_input, self.llm)
                #print(chatter_response)
                #self.psudo_type(chatter_response)
                #async for chunk in chatter_chain.astream({"user_input": user_input}):
                #    print(chunk, end="|", flush=True)
                self.old_df_markdown = self.markdown_df
                self.old_message_history = self.message_history
                self.message_history, code, self.markdown_df, explanation = model.run_model(model_input, self.llm, self.message_history, self.markdown_df, self, self.rerun)
                try:
                    os.remove(import_path_rerun_filename)
                except:
                    pass
                self.handle_response(output_path, code, explanation)
                
                #else:
                #    self.ai_response(chatter_response)

                # DOESNTWORK
                # Scroll to the bottom to see the latest messages
                self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())
        except:
            traceback.print_exc()
    

    def ai_response(self, ai_response="This is a placeholder response."):
        ai_label = QLabel(ai_response)
        ai_label.setAlignment(Qt.AlignmentFlag.AlignLeft) ##EA97FF
        ai_label.setStyleSheet("color: #bea5f0; background-color: #444444; padding: 5px; border-radius: 5px;font: 16px 'Ubuntu';")
        ai_label.setWordWrap(True)  # Enable word wrapping
        size_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        ai_label.setSizePolicy(size_policy)
        ai_label.setMaximumWidth(int((int(self.screen.width() * SCREENWIDTH)) * .45))
        ai_label.setFixedHeight(ai_label.sizeHint().height())
        self.chat_layout.addWidget(ai_label)

    
    def handle_response(self, output_path, code, explanation):
        try:
            print("RESULT PATH")
            print(output_path)
            self.data1_result = pd.read_csv(output_path, on_bad_lines='skip')
            self.rerun = True
            print('Rerun is True')
            self.message_history = ChatPromptTemplate.from_messages([SystemMessage(content=self.system_prompt)])
            self.display_result_data(self.data1_result)
            self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum()) 
            self.psudo_type(explanation)           
        except Exception as e:
            print("ERROR IN GUI - Rs")
            exc_type, exc_obj, exc_tb = sys.exc_info()
            traceback.print_exc()
            #model.rerun_after_error(code, exc_obj)


def check_machine_competence():
    if multiprocessing.cpu_count() < 16:
        return False
    else:
        return True
    
    
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
        print('t-dir', base_path)
    except Exception:
        base_path = os.path.abspath(".")
        print('exceptdtem dir')
        print(base_path)
    return os.path.join(base_path, relative_path)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    chat_interface = ChatInterface()
    chat_interface.show()
    

    sys.exit(app.exec())
