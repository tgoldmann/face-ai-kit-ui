from PyQt5.QtWidgets import QDialog, QProgressBar, QTextEdit, QVBoxLayout

class ProgressWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Processing...")
        self.setGeometry(150, 150, 500, 300)

        self.layout = QVBoxLayout()
        self.progress_bar = QProgressBar(self)
        self.layout.addWidget(self.progress_bar)

        self.console = QTextEdit(self)
        self.console.setReadOnly(True)
        self.layout.addWidget(self.console)

        self.setLayout(self.layout)

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def append_console(self, text):
        self.console.append(text)