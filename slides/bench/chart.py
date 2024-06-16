import csv
import os
import re
import matplotlib.pyplot as plt

if not os.path.exists('result.csv'):
    os.system('build/main --benchmark_format=csv > result.csv')

# read result.csv and plot it
with open('result.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader) # skip header row
    x = []
    y = []
    for row in csv_reader:
        name = row[0]
        # name format: BM_latency/offset:58/min_time:0.050, we need the offset only
        match = re.search(r'/offset:(\d+)', name)
        if match:
            offset = int(match.group(1))
            x.append(offset)
            cpu_time = float(row[3])
            y.append(cpu_time)

plt.bar(x, y)
plt.xlabel('Offset (bytes)')
plt.ylabel('CPU Time (ns)')
plt.title('Benchmark Result')
plt.show()
