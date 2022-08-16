#!/usr/bin/env python

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *


import threading
import requests
import json
import time
import sys
import os

def get_messages():
    roomid = 14248205
    url = f'https://api.live.bilibili.com/xlive/web-room/v1/dM/gethistory?roomid={roomid}'
    req = requests.get(url)
    data = json.loads(req.text)
    msgs = data['data']['room']
    res = []
    for msg in msgs:
        user = msg['nickname']
        text = msg['text']
        res.append(f'{user}: {text}')
    return res



class MainWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowOpacity(0.85)
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setWindowFlag(Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowTitle('双笙子佯谬的弹幕姬')
        self.setGeometry(0, 0, 300, 200)

        self.lv = QListView()
        self.lv.setSelectionMode(QAbstractItemView.NoSelection)
        self.slm = QStringListModel()
        self.slm.setStringList(['(没有弹幕)'])
        self.lv.setModel(self.slm)

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.lv)
        self.setLayout(self.layout)

        self.update()

    def update(self):
        threading.Timer(5, self.update).start()
        msgs = get_messages()
        # print('update got:', msgs)
        self.slm.setStringList(msgs)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
