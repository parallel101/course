from rich.console import Console
from rich.syntax import Syntax
import sys
import time
import os

with open(os.path.join(sys.argv[1], 'code'), 'r') as f:
    my_code = f.read()
syntax = Syntax(my_code, sys.argv[2], theme=sys.argv[3], line_numbers=True)
console = Console()
console.print(syntax)

time.sleep(0.04)
with open(os.path.join(sys.argv[1], 'done.lock'), 'w') as f:
    f.write('ok')
while not os.path.exists(os.path.join(sys.argv[1], 'exit.lock')):
    time.sleep(0.04)
