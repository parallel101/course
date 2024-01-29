#!/usr/bin/python

import requests
import json
import time
import os

with open(os.path.expanduser('~/.bilibili-cookies.json'), 'r') as f:
    cookies = {item['name']: item['value'] for item in json.load(f)}

headers = {
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.5',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.116 Safari/537.36'
}

roomid = 14248205
url = f'https://api.live.bilibili.com/xlive/web-room/v1/dM/gethistory?roomid={roomid}'
while True:
    req = requests.get(url, headers=headers, cookies=cookies)
    data = json.loads(req.text)
    msgs = data['data']['room']
    with open('/tmp/danmu.txt', 'w') as f:
        res = '\033[H\033[2J\033[3J'
        for msg in msgs:
            user = msg['nickname']
            text = msg['text']
            f.write(f'{user}: {text}\n')
            res += f'\033[31;1m{user}\033[0m: \033[32;1m{text}\033[0m\n'
        print(res, end='')
    time.sleep(10)
