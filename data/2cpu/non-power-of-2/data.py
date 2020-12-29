import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_algo(filename):
    x = []
    y = []

    with open(filename) as f:
        for line in f:
            num = line.split()
            x.append(num[0])
            y.append(num[1])

    x = np.array(x)
    y = np.array(y)

    return x.astype('float'), y.astype('float')

x, y1 = read_algo('algo1.txt')
_, y2 = read_algo('algo2.txt')
_, y3 = read_algo('algo3.txt')
_, y4 = read_algo('algo4.txt')
_, y5 = read_algo('algo5.txt')
_, y6 = read_algo('algo6.txt')

df = pd.DataFrame({'x':x, 'y1': y1, 'y2': y2, 'y3': y3, 'y4': y4, 'y5': y5, 'y6': y6})

print(df.head())

# style
plt.style.use('seaborn-darkgrid')
 
# create a color palette
palette = plt.get_cmap('Set1')

# multiple line plot
num=0
for column in df.drop('x', axis=1):
    num+=1
    plt.plot(df['x'], df[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)

# plt.axis('square')

# print(y1,y2,y3,y4,y5,y6, sep='\n')
# 
# plt.plot(x, y1, label='algo1')
# plt.plot(x, y2, label='algo2')
# plt.plot(x, y3, label='algo3')
# plt.plot(x, y4, label='algo4')
# plt.plot(x, y5, label='algo5')
# plt.plot(x, y6, label='algo6')
# fig, ax = plt.subplots(1, figsize=(8, 6))
# fig.suptitle('Multiple Lines in Same Plot', fontsize=15)
# ax.plot(x, y1, color="red", label="algo1")
# ax.plot(x, y2, color="green", label="algo2")
# ax.plot(x, y3, color="blue", label="algo3")
# plt.legend(loc="lower right", title="Legend Title", frameon=False)
# plt.axis('equal')
# plt.axis('scaled')



plt.legend()
plt.show()
# plt.savefig('2cpu-non2.png')
