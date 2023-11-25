from PyQt5 import QtWidgets, uic
import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import transformers


class AppealWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(AppealWindow, self).__init__()
        self.init_ui()

    def init_ui(self):
        uic.loadUi('form.ui', self)

        self.act_exit.triggered.connect(self.close)
        self.act_load.triggered.connect(self.load)
        self.act_clear_msg.triggered.connect(self.clear_messages)
        self.act_versions.triggered.connect(self.show_versions)

        self.btn_clear_input.clicked.connect(self.clear_input)
        self.btn_analysis.clicked.connect(self.analysis)

        self.load_model(self)

        self.show()

    def analysis(self):
        pass

    def load(self):
        # choose file
        filepath = None
        self.load_model(filepath)

    def load_model(self, filepath=None):
        self.pte_messages.append('Загрузка модели...')
        if filepath is None:
            filepath = 'appeal'

        self.sbert_t = transformers.AutoTokenizer.from_pretrained("ai-forever/sbert_large_mt_nlu_ru")
        self.sbert_m = transformers.AutoModel.from_pretrained("ai-forever/sbert_large_mt_nlu_ru")

        path = Path(filepath)
        with open(path/'theme_dict.pickle', rb) as f:
            self.themes = pickle.load(f)

        # bert, kmean, kmean_c, cc, cc_c
        self.pte_messages.append('Готово')

    def clear_messages(self):
        self.pte_messages.clear()

    def clear_input(self):
        self.pte_input.clear()

    def show_versions(self) -> None:
        import transformers
        from PyQt5.QtCore import PYQT_VERSION_STR

        libraries = [np, pd, transformers]
        self.pte_messages.append('    Использованные библиотеки:')
        self.pte_messages.append('Python ' + sys.version)
        self.pte_messages.append('PyQt5 ' + PYQT_VERSION_STR)
        for library in libraries:
            self.pte_messages.append(library.__name__ + ' ' + library.__version__)


app = QtWidgets.QApplication(sys.argv)
window = AppealWindow()
app.exec_()
