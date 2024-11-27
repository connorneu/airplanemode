from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel, QScrollArea, QFileDialog, QTableWidget, QTableWidgetItem, QSizePolicy
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QPixmap
import pandas as pd
import sys
import model
import os
from dateutil.parser import parse
import traceback
import ast
import csv


SCREENWIDTH = 0.5
SCREENHEIGHT = 0.7

class ChatInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chat Interface")
        self.screen = QApplication.primaryScreen().size()
        self.setGeometry(100, 100, int(self.screen.width() * SCREENWIDTH), int(self.screen.height() * SCREENHEIGHT))

        #self.setGeometry(300, 100, 600, 500)

        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout for all chat messages
        self.chat_layout = QVBoxLayout()
        self.chat_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        ai_response = "Upload your data to begin."  # Replace with actual response logic
        ai_label = QLabel(f"AI: {ai_response}")
        ai_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        ai_label.setStyleSheet("color: #ffcc00; background-color: #444444; padding: 5px;"
                               "border-radius: 5px;font: 16px 'Ubuntu';")
        ai_label.setFixedHeight(ai_label.sizeHint().height())
        ai_label.setFixedWidth(ai_label.sizeHint().width())
        self.chat_layout.addWidget(ai_label)
        
        # Scroll area to hold chat layout
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        chat_container = QWidget()
        chat_container.setLayout(self.chat_layout)
        self.scroll_area.setWidget(chat_container)
        
        # File drop area setup
        self.file_drop_area = QLabel("Click here or drag and drop a file to start", self)
        self.file_drop_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_drop_area.setStyleSheet("border: 2px dashed #888; padding: 20px; font: 20px 'Ubuntu'; color: #888;")
        self.file_drop_area.setAcceptDrops(True)
        
        # Input box and submit button setup (initially hidden)
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Ask a question...")
        self.input_box.returnPressed.connect(self.handle_submit)
        self.input_box.setVisible(False)
        
        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.handle_submit)
        self.submit_button.setVisible(False)
        
        # Additional file upload icon
        self.upload_icon = QPushButton()
        self.upload_icon.setIcon(QIcon("upload_icon.png"))  # Replace with path to an upload icon image
        self.upload_icon.setIconSize(QPixmap("upload_icon.png").size())
        self.upload_icon.clicked.connect(self.upload_additional_file)
        self.upload_icon.setVisible(False)

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

        self.data1 = None
        self.data1_col = None
        self.data1_trunc = None
        self.data1_result = None
        self.data1_filepath = None
        self.client = model.build_client()
        self.message_history = []
        self.docsplits = None
        self.embeddings = None
        self.llm = None
        self.retriever = None
        self.work_dir = None

        # create directory for working files
        if not os.path.exists('work'):
            os.makedirs('work')
        self.work_dir = os.path.join(os.getcwd(), 'work')


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


    def update_column_types(self, df):
        for col in df.columns:
            try:
                df[col+'_t'] = df[col].astype(str).apply(self.is_date)
                if df[col+'_t'].any():
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    df.drop(col+'_t', axis=1, inplace=True)
                df.drop(col+'_t', axis=1, inplace=True)
            except:
                pass
        return df


    def is_date(self, string, fuzzy=False):
        """
        Return whether the string can be interpreted as a date.

        :param string: str, string to check for date
        :param fuzzy: bool, ignore unknown tokens in string if True
        """
        try: 
            if '-' in string or '/' in string or  '\\' in string or ' ' in string:
                parse(string, fuzzy=fuzzy)
                return True
            else:
                return False

        except ValueError:
            return False
    
    
    def read_file(self, filepath):
        if '.xl' in filepath:
            return pd.read_excel(filepath) 
        elif '.csv' in filepath:
            return pd.read_csv(filepath)
        elif '.txt' in filepath:
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(filepath)
            print('dialect', dialect)
            return pd.read_csv(filepath, sep=dialect) #untested
        else:
            print('Unsuported file type')
            return 'Unsuported file type'


    def trunc_data(self, data):
        orig_len = len(data.to_string())
        if orig_len < 3000:
            return data
        else:
            cut_x = 10000
            curr_len = orig_len
            while curr_len > 3000:
                data = data.head(cut_x)
                curr_len = len(data.to_string())
                if curr_len < 3000:
                    return data
                else:
                    if len(str(cut_x)) >= 4:
                        cut_x -= 1000
                    else:
                        cut_x -= 100


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
            # read data
            data = self.read_file()
            self.data1 = data
            path, filename = os.path.split(file_path)
            self.data1_filepath = os.path.join(self.work_dir, filename)
            # truncate data to generate embeddings
            self.data1_trunc  = self.trunc_data(self.data1_filename)
            self.data1_col = list(self.data1.columns)               
            suggestions, self.docsplits, self.embeddings, self.llm, self.retriever = model.suggest_actions(self.data1_trunc)
            #try:
            suggestions = ast.literal_eval(suggestions)
            suggestions = [n.strip() for n in suggestions]
            #except:
            #    suggestions = ['Calculate the difference between dates', 
            #                   'Filter your dataset to based on complicated requirements',
            #                   'Discover relationships between different columns']
            self.display_data_preview(data)
            self.ai_response("Nice Data! Here are some suggestions of what kind of analysis you can do:")
            #self.display_suggestion_buttons(suggestions)
            self.ai_response(suggestions)
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
            button.setStyleSheet("background-color: #444444; color: #ffffff; padding: 10px; border-radius: 5px;")
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

        button.setStyleSheet("background-color: #444444; color: #ffffff; padding: 10px; border-radius: 5px;")

        button_container = QWidget()
        button_container.setLayout(button_layout)
        self.chat_layout.addWidget(button_container, alignment=Qt.AlignmentFlag.AlignLeft)


    def handle_suggestion_click(self, suggestion):
        # Display the selected suggestion as a user input
        user_label = QLabel(f"User selected: {suggestion}")
        user_label.setStyleSheet(
            "color: #00ffcc; background-color: #333333; padding: 5px; border-radius: 5px; font: 16px 'Ubuntu';"
        )
        user_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.chat_layout.addWidget(user_label)

        # Trigger the model based on the suggestion
        model_input = {"import_file": self.data1_filepath, "user_input": suggestion, "column_headers": self.data1_col}
        model.run_model(self.client, model_input, self.data1.to_string())
        result_data = pd.read_csv('doData_Output.csv')
        self.display_result_data(result_data)
        self.ai_response(f"Results for: {suggestion}")


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


        # Create a download button
        download_button = QPushButton()
        download_button.setIcon(QIcon("download_icon.png"))  # Replace with the path to your download icon
        download_button.setToolTip("Download Table")
        download_button.clicked.connect(lambda: self.download_table(data))


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

            if not os.path.isfile('doData_Output.csv'):
                import_file = self.data1_filepath
            else:
                is_redo = model.new_or_old(self.client, user_input)
                print('IS REDO', is_redo)
                if is_redo:
                    import_file = self.data1_filepath
                else:
                    import_file = 'doData_Output.csv'
                    self.data1_result.to_csv(self.data1_filepath)

            column_headers = self.data1_col
            model_input = {"import_file":import_file, "user_input":user_input, "column_headers": column_headers}
            self.message_history = model.run_model(self.data1, model_input, self.docsplits, self.embeddings, self.llm, self.retriever)
            self.ai_response('Here is your result so far')
            self.data1_result = pd.read_csv('doData_Output.csv')
            self.display_result_data(self.data1_result)

            # Scroll to the bottom to see the latest messages
            self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())
    

    def ai_response(self, ai_response="This is a placeholder response."):
        ai_label = QLabel(f"AI: {ai_response}")
        ai_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        ai_label.setStyleSheet("color: #ffcc00; background-color: #444444; padding: 5px; border-radius: 5px;font: 16px 'Ubuntu';")
        ai_label.setWordWrap(True)  # Enable word wrapping
        size_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        ai_label.setSizePolicy(size_policy)
        ai_label.setMaximumWidth(int((int(self.screen.width() * SCREENWIDTH)) * .45))
        ai_label.setFixedHeight(ai_label.sizeHint().height())
        self.chat_layout.addWidget(ai_label)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    chat_interface = ChatInterface()
    chat_interface.show()
    sys.exit(app.exec())
