#!/usr/bin/python

from selenium import webdriver
import time, re, os

url = 'https://live.bilibili.com/2171135'

opts = webdriver.ChromeOptions()
opts.add_argument('headless')
wd = webdriver.Chrome(options=opts)
wd.implicitly_wait(10)
wd.get(url)

while True:
    res = []
    for e in wd.find_elements_by_css_selector('div.chat-item.danmaku-item'):
        usr = e.get_attribute('data-uname')
        msg = e.get_attribute('data-danmaku')
        res.append((usr, msg))
    os.system('clear')
    for usr, msg in res:
        print(f'{usr}: {msg}')
    time.sleep(8)
