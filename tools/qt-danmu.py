#!/usr/bin/env python

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

import subprocess
import traceback
import threading
import tempfile
import requests
import queue
import json
import time
import sys
import os
import re

# with open(os.path.expanduser('~/.bilibili-cookies.json'), 'r') as f:
#     cookies = {item['name']: item['value'] for item in json.load(f)}

headers = {
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.5',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.116 Safari/537.36'
}
cookies = {}

if os.path.exists('.bilibili-options.json'):
    with open('.bilibili-options.json', 'r') as f:
        options = json.load(f)
else:
    options = {}

if len(options) == 0:
    options = {
        'width': 420,
        'height': 120,
        'refreshInterval': 6,
        'backgroundOpacity': 0.15,
        'foregroundOpacity': 0.35,
        'fontColor': 'black',
        'windowLocation': 'bottomLeft',
        'showMusicName': True,
        'danmuFile': tempfile.gettempdir() + '/danmu.txt',
        'danmuFormat': '{danmu}\n[B站弹幕有屏蔽词，没显示就是叔叔屏蔽了]\n[已知屏蔽词：小彭老师、皇帝卡、Electron]',
    }

def current_music():
    if sys.platform != 'linux':
        return ''
    try:
        suffix = r'( - VLC media player|_哔哩哔哩_bilibili — Mozilla Firefox)$'
        with subprocess.Popen(['wmctrl', '-l'], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
            stdout, _ = p.communicate()
            stdout = stdout.decode()
            title = ''
            for line in stdout.splitlines():
                # 0x02000003  3 archer 终端
                m = re.match(r'^0x([0-9a-f]+)\s+(\d+)\s+(.*?)\s+(.*)$', line)
                if m:
                    title = m.group(4).strip()
                if re.search(suffix, title):
                    break
            else:
                title = ''
        if title:
            title = re.sub(suffix, '', title)
        if title.endswith('.mp4'):
            title = title[:-len('.mp4')]
        return title
    except:
        traceback.print_exc()
        return ''

# def login():
#     url = 'https://passport.bilibili.com/x/passport-login/web/qrcode/generate'
#     req = requests.get(url, headers=headers)
#     cookies.update(req.cookies.get_dict())
#     data = json.loads(req.text)
#     qr_url = data['data']['url']
#     qrcode_key = data['data']['qrcode_key']
#     print(qrcode_key, qr_url)
#     import qrcode
#     img = qrcode.make(qr_url)
#     img.show()
#     while True:
#         url = f'https://passport.bilibili.com/x/passport-login/web/qrcode/poll?qrcode_key={qrcode_key}'
#         req = requests.get(url, headers=headers, cookies=cookies)
#         print(req.text, req.cookies.get_dict())
#         cookies.update(req.cookies.get_dict())
#         data = json.loads(req.text)
#         data = data['data']
#         if data['code'] == 0:
#             print('登录成功')
#             break
#         elif data['code'] == 86038:
#             print('二维码已过期')
#             break
#         elif data['code'] == 86090:
#             print('请在手机App上确认登录')
#         elif data['code'] == 86101:
#             print('请在手机App上扫描二维码')
#         else:
#             print(data['message'])
#         time.sleep(3)
#     with open('.bilibili-cookies.json', 'w') as f:
#         json.dump(cookies, f)

class MyThread(QThread):
    def __init__(self, parent, func, *args, **kwargs):
        super().__init__(parent)
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        self.func(*self.args, **self.kwargs)

class LoginWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle('登录')
        self.resize(400, 400)

        self.image = QLabel()
        self.image.setScaledContents(True)
        self.image.setPixmap(QPixmap('icon.png').scaled(400, 400, Qt.KeepAspectRatio))

        self.label = QLabel('欢迎使用弹幕助手')

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.image)
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.qr_url = None
        self.qrcode_key = None
        self.status = None
        self.succeeded = False

    def login(self):
        self.clear_login()
        t = MyThread(self, self.get_qrcode)
        t.finished.connect(self.show_qrcode)
        t.start()
        self.label.setText('正在生成二维码...')

    def clear_login(self):
        self.qr_url = None
        self.qrcode_key = None
        self.status = None
        self.succeeded = False
        self.image.setPixmap(QPixmap('icon.png').scaled(400, 400, Qt.KeepAspectRatio))

    def get_qrcode(self):
        url = 'https://passport.bilibili.com/x/passport-login/web/qrcode/generate'
        req = requests.get(url, headers=headers)
        cookies.update(req.cookies.get_dict())
        data = json.loads(req.text)
        self.qr_url = data['data']['url']
        self.qrcode_key = data['data']['qrcode_key']
        self.status = '请在手机App上扫描二维码'

    def show_qrcode(self):
        if self.qr_url is not None:
            # print(self.qr_url)
            import qrcode
            img = qrcode.make(self.qr_url)
            img = img.convert('RGBA')
            img = QImage(img.tobytes(), img.width, img.height, QImage.Format_RGBA8888)
            self.image.setPixmap(QPixmap.fromImage(img).scaled(400, 400, Qt.KeepAspectRatio))
            self.qr_url = None
        if self.status:
            self.label.setText(self.status)
        if self.succeeded:
            self.clear_login()
        if self.isHidden() or self.succeeded:
            return
        t = MyThread(self, self.poll_qrcode)
        t.finished.connect(self.show_qrcode)
        t.start()

    def poll_qrcode(self):
        time.sleep(3)
        url = f'https://passport.bilibili.com/x/passport-login/web/qrcode/poll?qrcode_key={self.qrcode_key}'
        req = requests.get(url, headers=headers, cookies=cookies)
        # print(req.text, req.cookies.get_dict())
        cookies.update(req.cookies.get_dict())
        data = json.loads(req.text)
        data = data['data']
        if data['code'] == 0:
            with open('.bilibili-cookies.json', 'w') as f:
                json.dump(cookies, f)
            del options['roomId']
            with open('.bilibili-options.json', 'w') as f:
                json.dump(options, f)
            self.status = '登录成功'
            self.succeeded = True
        elif data['code'] == 86038:
            self.status = '二维码已过期'
            self.succeeded = True
        elif data['code'] == 86090:
            self.status = '请在手机App上确认登录'
        elif data['code'] == 86101:
            self.status = '请在手机App上扫描二维码'
        else:
            self.status = data['message']

def get_messages():
    if len(cookies) == 0:
        if os.path.exists('.bilibili-cookies.json'):
            with open('.bilibili-cookies.json', 'r') as f:
                cookies.update(json.load(f))
    if len(cookies) == 0:
        return ['(未登录，请先右键托盘图标，在设置中扫码登录您的B站账号)']
    if 'roomId' not in options:
        url = 'https://api.bilibili.com/x/web-interface/nav'
        req = requests.get(url, headers=headers, cookies=cookies)
        mid = json.loads(req.text)['data']['mid']
        url = f'https://api.live.bilibili.com/room/v1/Room/getRoomInfoOld?mid={mid}'
        req = requests.get(url, headers=headers, cookies=cookies)
        data = json.loads(req.text)['data']
        roomid = data['roomid']
        # title = data['title']
        # print(f'已进入直播间 {title} ({roomid})')
        set_option('roomId', roomid)
    else:
        roomid = options['roomId']
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

def set_option(key, value):
    options[key] = value
    with open('.bilibili-options.json', 'w') as f:
        json.dump(options, f)

class SettingsWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        icon = QIcon('icon.png')
        self.setWindowIcon(icon)
        self.setWindowTitle('弹幕助手')
        self.resize(400, 400)

        self.login_window = LoginWindow()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.login_window)
        hlayout = QHBoxLayout()
        button = QPushButton('登录')
        button.clicked.connect(self.login_window.login)
        hlayout.addWidget(button)
        button = QPushButton('退出')
        button.clicked.connect(QApplication.quit)
        hlayout.addWidget(button)
        button = QPushButton('重启')
        button.clicked.connect(self.restart)
        hlayout.addWidget(button)
        layout.addLayout(hlayout)
        layout.addWidget(QLabel('窗口大小(宽x高)'))
        self.width = QSpinBox()
        self.width.setRange(100, 1000)
        self.width.setValue(options['width'])
        self.width.valueChanged.connect(lambda value: set_option('width', value))
        self.height = QSpinBox()
        self.height.setRange(100, 1000)
        self.height.setValue(options['height'])
        self.height.valueChanged.connect(lambda value: set_option('height', value))
        layout.addWidget(self.width)
        layout.addWidget(self.height)
        layout.addWidget(QLabel('刷新间隔(秒)'))
        self.refreshInterval = QSpinBox()
        self.refreshInterval.setRange(1, 60)
        self.refreshInterval.setValue(options['refreshInterval'])
        self.refreshInterval.valueChanged.connect(lambda value: set_option('refreshInterval', value))
        layout.addWidget(self.refreshInterval)
        layout.addWidget(QLabel('背景透明度(%)'))
        self.backgroundOpacity = QSpinBox()
        self.backgroundOpacity.setRange(0, 100)
        self.backgroundOpacity.setValue(int(options['backgroundOpacity'] * 100))
        self.backgroundOpacity.valueChanged.connect(lambda value: set_option('backgroundOpacity', value / 100))
        layout.addWidget(self.backgroundOpacity)
        layout.addWidget(QLabel('前景透明度(%)'))
        self.foregroundOpacity = QSpinBox()
        self.foregroundOpacity.setRange(0, 100)
        self.foregroundOpacity.setValue(int(options['foregroundOpacity'] * 100))
        self.foregroundOpacity.valueChanged.connect(lambda value: set_option('foregroundOpacity', value / 100))
        self.showMusicName = QCheckBox('显示当前音乐')
        self.showMusicName.setChecked(options['showMusicName'])
        self.showMusicName.stateChanged.connect(lambda value: set_option('showMusicName', value))
        layout.addWidget(self.showMusicName)
        layout.addWidget(QLabel('字体颜色'))
        self.fontColor = QComboBox()
        self.fontColor.addItems(['black', 'white'])
        self.fontColor.setCurrentText(options['fontColor'])
        self.fontColor.currentTextChanged.connect(lambda value: set_option('fontColor', value))
        layout.addWidget(self.fontColor)
        layout.addWidget(QLabel('窗口位置'))
        self.windowLocation = QComboBox()
        self.windowLocation.addItems(['topLeft', 'topRight', 'bottomLeft', 'bottomRight'])
        self.windowLocation.setCurrentText(options['windowLocation'])
        self.windowLocation.currentTextChanged.connect(lambda value: set_option('windowLocation', value))
        layout.addWidget(self.windowLocation)
        layout.addWidget(QLabel('弹幕输出文件(可供OBS等软件使用)'))
        self.danmuFile = QLineEdit()
        self.danmuFile.setText(options['danmuFile'])
        self.danmuFile.textChanged.connect(lambda value: set_option('danmuFile', value))
        layout.addWidget(self.danmuFile)
        layout.addWidget(QLabel('弹幕文本输出格式'))
        self.danmuFormat = QTextEdit()
        self.danmuFormat.setAcceptRichText(False)
        self.danmuFormat.setFixedHeight(60)
        self.danmuFormat.setText(options['danmuFormat'])
        self.danmuFormat.textChanged.connect(lambda: set_option('danmuFormat', self.danmuFormat.toPlainText()))
        layout.addWidget(self.danmuFormat)
        self.setLayout(layout)

    def restart(self):
        os.execl(sys.executable, sys.executable, *sys.argv)

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

        self.settings_window = SettingsWindow()

        tray_icon = QSystemTrayIcon(self)
        tray_icon.setIcon(icon)
        tray_icon.setToolTip('弹幕助手')

        tray_menu = QMenu(self)
        action = tray_menu.addAction("显示/隐藏")
        action.triggered.connect(self.toggle_window)
        action = tray_menu.addAction("设置/登录")
        action.triggered.connect(self.settings_window.show)
        action = tray_menu.addAction("退出")
        action.triggered.connect(QApplication.quit)

        tray_icon.setContextMenu(tray_menu)
        tray_icon.activated.connect(self.toggle_window)
        tray_icon.show()

        w, h = options['width'], options['height']
        self.resize(w, h)

        desktop = QApplication.desktop()
        if options['windowLocation'] == 'bottomRight':
            bottom_right = desktop.screenGeometry().bottomRight()
            self.move(bottom_right - QPoint(w, h))
        elif options['windowLocation'] == 'bottomLeft':
            bottom_left = desktop.screenGeometry().bottomLeft()
            self.move(bottom_left - QPoint(0, h))
        elif options['windowLocation'] == 'topRight':
            top_right = desktop.screenGeometry().topRight()
            self.move(top_right - QPoint(w, 0))
        elif options['windowLocation'] == 'topLeft':
            top_left = desktop.screenGeometry().topLeft()
            self.move(top_left)

        bga, fga = options['backgroundOpacity'], options['foregroundOpacity']
        bgc, fgc = {
            'black': (255, 0),
            'white': (0, 255),
        }[options['fontColor']]
        self.setStyleSheet(f'* {{font-size: 18px; color: rgba({fgc}, {fgc}, {fgc}, {fga}); background-color: rgba({bgc}, {bgc}, {bgc}, {bga}); border-radius: 10px; padding: 0px;}}')

        self.lv = QListView()
        self.lv.setSelectionMode(QAbstractItemView.NoSelection)
        self.slm = QStringListModel()
        self.slm.setStringList(['(请稍等)'])
        self.lv.setModel(self.slm)

        self.lv.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.lv.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.lv)
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
            if self.queue.full():
                time.sleep(1)
                continue
            try:
                msgs = get_messages()
            except:
                traceback.print_exc()
                continue
            if len(msgs) == 0:
                msgs = ['(还没有弹幕，快来发一条吧)']
            if options['showMusicName']:
                music = current_music().strip()
                if music:
                    music = '当前播放：' + music
                    msgs.append(music)
            if options['danmuFile']:
                with open(options['danmuFile'], 'w') as f:
                    fmt = options['danmuFormat']
                    danmu = '\n'.join(msgs)
                    if fmt:
                        danmu = fmt.format(danmu=danmu)
                    f.write(danmu)
            try:
                self.queue.put(msgs, block=False)
            except queue.Full:
                pass
            time.sleep(options['refreshInterval'])

    def update_messages(self):
        if self.isHidden():
            return
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
    # print(current_music())
    sys.exit(main())
