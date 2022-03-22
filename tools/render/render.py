#from rich.console import Console
#from rich.syntax import Syntax
import subprocess
import sys
import time
import os

p = subprocess.Popen(['vim', '-R', '-n', sys.argv[1]])

time.sleep(1)
with open(os.path.join(sys.argv[2], 'done.lock'), 'w') as f:
    f.write('ok')
while not os.path.exists(os.path.join(sys.argv[2], 'exit.lock')):
    time.sleep(0.04)
p.kill()
