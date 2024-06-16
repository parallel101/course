import csv
import os
import re
import sys
import matplotlib.pyplot as plt

key = sys.argv[1]

if not os.path.exists(key + '.csv'):
    os.system('build/main --benchmark_filter=' + key + ' --benchmark_format=csv | tee ' + key + '.csv')

# read result.csv and plot it
with open(key + '.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader) # skip header row
    x = []
    y = []
    for row in csv_reader:
        name = row[0]
        # name format: BM_latency/offset:58/min_time:0.050, we need the offset only
        match = re.search(key + r'/offset:(\d+)', name)
        if match:
            offset = int(match.group(1))
            x.append(offset)
            cpu_time = float(row[3])
            y.append(cpu_time)

bar = plt.bar(x, y)
for i in range(len(bar)):
    if i % 8 != 0:
        bar[i].set_color('#1fb477')
plt.xlabel('Offset (bytes)')
plt.ylabel('CPU Time (ns)')
plt.title(key)
plt.show()
