#!/usr/bin/env python3

import matplotlib.pyplot as plt
import subprocess as sp

basampshift = 18
basenshift = 9
nshifts = 12

x0 = []
ys = {}

for i in range(nshifts):
    n = 1 << (basenshift + i)
    samps = 1 << (basampshift - i)
    x0.append(n)

    args = ['-DTEST_SIZE={}'.format(n), '-DTEST_SAMPLES={}'.format(samps)]
    sp.check_call(['cmake', '-B', 'build'] + args)
    sp.check_call(['cmake', '--build', 'build'])
    ret = sp.check_output(['build/testbench']).decode()

    for line in ret.splitlines():
        key, val = line.split(': ', 2)
        val = int(val)
        ys.setdefault(key, []).append(val)

for k, y in ys.items():
    plt.plot(x0, y, label=k)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('array size')
plt.ylabel('time (ns)')
plt.title('different data layout')
plt.legend()
plt.show()
