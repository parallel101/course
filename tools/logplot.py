#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

xs = np.array([16, 128, 1024, 16*1024, 128*1024, 1024*1024])
xs = xs / 1024
ys = np.array(list(reversed([71526565, 8643359, 742196, 25630, 2361, 194])))
ys = ys / 1000 / 1000 / 1000
ys = xs / ys * 2

plt.plot(xs, ys)

plt.xscale('log')
#plt.yscale('log')
plt.xlabel('data size (MB)')
plt.ylabel('bandwidth (MB/s)')
plt.title('different data size')
plt.legend()
plt.show()
