import subprocess
import sys
import time
import os

p = subprocess.Popen(['vim', '-R', '-n', '-i', 'NONE', sys.argv[1],
    '+:set signcolumn=no\n:call timer_start(50, ' +
    '{-> execute("w! ' + os.path.join(sys.argv[2], 'done.lock') + '")})'])
while not os.path.exists(os.path.join(sys.argv[2], 'exit.lock')):
    time.sleep(0.04)
p.kill()
