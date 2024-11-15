from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel, QScrollArea, QFileDialog, QTableWidget, QTableWidgetItem
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QPixmap
import pandas as pd
import sys
import model

class ChatInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chat Interface")
        self.screen = QApplication.primaryScreen().size()
        self.setGeometry(100, 100, int(self.screen.width() * 0.5), int(self.screen.height() * 0.8))

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
        self.data2 = None
        self.data3 = None
        self.data4 = None
        self.numdata = 0
        self.datas = [self.data1, self.data2, self.data3, self.data4]
        self.client = model.build_client()

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

    def handle_file_upload(self, file_path):
        # Hide the file drop area and show input box and upload icon
        self.file_drop_area.setVisible(False)
        self.input_box.setVisible(True)
        self.submit_button.setVisible(True)
        self.upload_icon.setVisible(True)
        
        # Display the uploaded filename in a banner on the left side of the response area
        file_label = QLabel(f"File uploaded: {file_path}")
        file_label.setStyleSheet("color: #ffffff; background-color: #555555; padding: 10px; border-radius: 5px;")
        file_label.setFixedHeight(file_label.sizeHint().height())
        file_label.setFixedWidth(file_label.sizeHint().width())
        self.chat_layout.addWidget(file_label, alignment= Qt.AlignmentFlag.AlignRight)
        
        # Display a preview of the file data in a QTableWidget if it’s a CSV
        try:
            data = pd.read_csv(file_path)  # Adjust for other file types if needed
            if self.numdata == 0:
                self.data1 = data
            elif self.numdata == 1:
                self.data2 = data
            elif self.numdata == 2:
                self.data3 = data
            elif self.numdata == 3:
                self.data4 = data
            self.numdata += 1
            self.display_data_preview(data)
            self.ai_response("Nice Data!")
        except Exception as e:
            # Display an error message if the file could not be read
            error_label = QLabel(f"Could not read file: {e}")
            error_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
            error_label.setStyleSheet("color: #ff0000; padding: 10px; border-radius: 5px;")
            self.chat_layout.addWidget(error_label)


    def display_data_preview(self, data):
        # Limit data preview to first 10 rows
        preview_data = data.head(50)
        
        # Create a QTableWidget for displaying the data preview
        table = QTableWidget(preview_data.shape[0], preview_data.shape[1])
        table.setHorizontalHeaderLabels(preview_data.columns)
        table.setVerticalHeaderLabels([str(i) for i in preview_data.index])

        # Set table width to 80% of the window width and align to the left
        print(self.chat_layout.contentsRect().width())
        table.setFixedWidth(int(self.screen.width() * 0.5) - 300)
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
        #table.setSizeAdjustPolicy(QTableWidget.SizeAdjustPolicy.AdjustToContents)

        # Align the table to the right in the chat layout
        table_container = QHBoxLayout()
        table_container.addWidget(table)
        table_container.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.chat_layout.addLayout(table_container)

    def upload_additional_file(self):
        # Open file dialog for additional file upload
        self.open_file_dialog()

    def handle_submit(self):
        # Get user input and display it on the right side
        user_input = self.input_box.text()
        if user_input.strip():  # Check if there's input
            # Display user input on the right
            user_label = QLabel(f"User: {user_input}")
            user_label.setStyleSheet("color: #00ffcc; background-color: #333333; padding: 5px; border-radius: 5px;font: 16px 'Ubuntu';")
            user_label.setFixedHeight(user_label.sizeHint().height())
            user_label.setFixedWidth(user_label.sizeHint().width())
            user_label.setMaximumWidth(2000)
            user_label.setMinimumHeight(20)
            user_label.setMaximumHeight(2000)
            self.chat_layout.addWidget(user_label, alignment= Qt.AlignmentFlag.AlignRight)
            
            # Clear the input box for new input
            self.input_box.clear()

            # run the model
            #model.run_model(user_input, self.client, self.data1.to_string())
            
            # Generate and display AI response on the left
            ai_response = "This is a placeholder response."  # Replace with actual response logic
            ai_label = QLabel(f"AI: {ai_response}")
            ai_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
            ai_label.setStyleSheet("color: #ffcc00; background-color: #444444; padding: 5px; border-radius: 5px;font: 16px 'Ubuntu';")
            ai_label.setFixedHeight(ai_label.sizeHint().height())
            ai_label.setFixedWidth(ai_label.sizeHint().width())
            self.chat_layout.addWidget(ai_label)

            # Scroll to the bottom to see the latest messages
            self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())
    
    def ai_response(self, ai_response="This is a placeholder response."):
        ai_label = QLabel(f"AI: {ai_response}")
        ai_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        ai_label.setStyleSheet("color: #ffcc00; background-color: #444444; padding: 5px; border-radius: 5px;font: 16px 'Ubuntu';")
        ai_label.setFixedHeight(ai_label.sizeHint().height())
        ai_label.setFixedWidth(ai_label.sizeHint().width())
        self.chat_layout.addWidget(ai_label)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    chat_interface = ChatInterface()
    chat_interface.show()
    sys.exit(app.exec())
