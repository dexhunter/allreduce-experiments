import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

with open('2cpu-non2.txt') as f:
    contents = f.readlines()

x = []
y = []
for line in contents:
    num = line.split()
    x.append(num[0])
    y.append(num[1])

x = np.array(x)
y = np.array(y)

plt.plot(x, y)
plt.show()
plt.savefig('2cpu-non2.png')
