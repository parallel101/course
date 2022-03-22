#!/usr/bin/env python3

import subprocess
from PIL import ImageGrab
import time
import sys
import os
import tempfile
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('file', type=str)
ap.add_argument('--lang', type=str, default='auto')
ap.add_argument('--theme', type=str, default='monokai')
ap.add_argument('--show', action='store_true')
ap.add_argument('--out', type=str, default='')
ap = ap.parse_args()

my_lang = ap.lang
my_theme = ap.theme
my_file = ap.file
if my_file == '-':
    my_code = sys.stdin.read()
else:
    with open(ap.file, 'r') as f:
        my_code = f.read()
my_out = ap.out

with tempfile.TemporaryDirectory() as tmpdir:
    font_size = 14
    w, h = 0, 0
    for line in my_code.splitlines():
        w = max(w, len(line))
        h += 1
    w += 6
    h += 4
    with open(os.path.join(tmpdir, os.path.basename(my_file)), 'w') as f:
        f.write(my_code)
        f.write('\n')
    dirpath = os.path.dirname(os.path.abspath(__file__))
    p = subprocess.Popen(['xfce4-terminal', '--disable-server', '--title=term_to_screen_shot',
        '--geometry={}x{}+0+0'.format(w, h), '-x', sys.executable, os.path.join(dirpath, 'render.py'),
        os.path.join(tmpdir, os.path.basename(my_file)), tmpdir, my_lang, my_theme])
    while not os.path.exists(os.path.join(tmpdir, 'done.lock')):
        time.sleep(0.04)
    with open(os.path.join(tmpdir, 'exit.lock'), 'w') as f:
        f.write('ok')

    def getActiveWindowRect():
        res = subprocess.check_output(['bash', '-c', 'xwininfo -id $(xdotool getactivewindow)']).decode()
        out = {}
        for line in res.splitlines():
            line = line.split(':', 1)
            if len(line) >= 2:
                k, v = line
                k = k.strip()
                v = v.strip()
                out[k] = v
        bbox = out['Absolute upper-left X'], out['Absolute upper-left Y'], out['Width'], out['Height']
        return [int(x) for x in bbox]

    x, y, w, h = getActiveWindowRect()
    x += 1 + font_size // 2 * 3
    y += 1
    h -= font_size
    im = ImageGrab.grab((x, y, w, h))
    p.kill()

    if my_out:
        im.save(my_out)
        if ap.show:
            subprocess.check_call(['display', my_out])
    else:
        im.save(os.path.join(tmpdir, 'result.png'), format='png')
        if ap.show:
            subprocess.check_call(['display', os.path.join(tmpdir, 'result.png')])
        im.save(os.path.join(tmpdir, 'result.png'), format='png')
        subprocess.check_call(['xclip', '-selection', 'clipboard', '-t', 'image/png',
            '-i', os.path.join(tmpdir, 'result.png')])
