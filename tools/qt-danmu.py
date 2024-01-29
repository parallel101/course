#!/usr/bin/env python

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *


import threading
import requests
import queue
import json
import time
import sys
import os

with open(os.path.expanduser('~/.bilibili-cookies.json'), 'r') as f:
    cookies = {item['name']: item['value'] for item in json.load(f)}

headers = {
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.5',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.116 Safari/537.36'
}

def get_messages():
    roomid = 14248205
    url = f'https://api.live.bilibili.com/xlive/web-room/v1/dM/gethistory?roomid={roomid}'
    req = requests.get(url, headers=headers, cookies=cookies)
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

        icon = QIcon('icon.png')
        self.setWindowIcon(icon)
        self.setWindowTitle('弹幕助手')
        # self.setWindowOpacity(0.95)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.X11BypassWindowManagerHint)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

        tray_icon = QSystemTrayIcon(self)
        tray_icon.setIcon(icon)
        tray_icon.setToolTip('弹幕助手')

        # Create a menu for the tray icon
        tray_menu = QMenu(self)
        action = tray_menu.addAction("显示/隐藏")
        action.triggered.connect(self.toggle_window)
        action = tray_menu.addAction("退出")
        action.triggered.connect(QApplication.quit)

        w, h = 400, 200
        desktop = QApplication.desktop()
        bottom_right = desktop.screenGeometry().bottomRight()
        self.move(bottom_right - QPoint(w, h))
        self.resize(w, h)

        bg = 0.15
        fg = 0.55
        self.setStyleSheet('* {font-size: 18px; color: rgba(255, 255, 255, ' + str(fg) + '); background-color: rgba(0, 0, 0, ' + str(bg) + '); border-radius: 10px; padding: 0px;}')

        self.lv = QListView()
        self.lv.setSelectionMode(QAbstractItemView.NoSelection)
        self.slm = QStringListModel()
        self.slm.setStringList(['(没有弹幕)'])
        self.lv.setModel(self.slm)

        self.lv.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.lv.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.lv)

        # hlayout = QHBoxLayout()
        # button = QPushButton()
        # button.setText('退出')
        # button.clicked.connect(QApplication.quit)
        # hlayout.addWidget(button)
        #
        # button = QPushButton()
        # button.setText('退出')
        # button.clicked.connect(QApplication.quit)
        # hlayout.addWidget(button)
        #
        # layout.addLayout(hlayout)

        self.setLayout(layout)

        self.queue = queue.Queue(maxsize=1)

        self.update_messages()

        timer = QTimer(self)
        timer.timeout.connect(self.update_messages)
        timer.start(1000)
        timer.setSingleShot(False)
        
        threading.Thread(target=self.message_worker, daemon=True).start()

    def toggle_window(self):
        if self.isVisible():
            self.hide()
        else:
            self.show()

    def message_worker(self):
        while True:
            try:
                msgs = get_messages()
            except:
                import traceback
                traceback.print_exc()
                continue
            try:
                self.queue.put(msgs, block=False)
            except queue.Full:
                pass
            time.sleep(5)

    def update_messages(self):
        try:
            msgs = self.queue.get(block=False)
        except queue.Empty:
            return
        if self.slm.stringList() != msgs:
            self.slm.setStringList(msgs)
            self.lv.scrollToBottom()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
