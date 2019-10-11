import numpy as np
import math

ran = 1000
ran2 = math.sqrt(ran)
#m = 100
m = 1000
n = 3
sq = math.sqrt(ran)

def myRand(length, ran):
    return np.random.rand(length) * ran * 2 - ran

params = myRand(n+1, ran)

print(m)
print(n)

for i in range(m):
    x = np.append(myRand(n, ran2), 1)
    y = (x*params).sum()
    x += np.random.normal(0, 0.1, size = n+1)
    for xi in range(len(x)-1):
        print(x[xi])
    print(y)

for param in params:
    print(param)