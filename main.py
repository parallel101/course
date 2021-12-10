#!/usr/bin/env python

import subprocess
import threading
import tempfile
import json
import sys
import os

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *


class MainWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowOpacity(0.85)
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setWindowTitle('双笙子佯谬的弹幕姬')
        self.setGeometry(750, 200, 480, 640)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
