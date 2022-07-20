# %%
file = open('1.txt', 'r')
Lines = file.readlines()
file.close()
file = open('1.txt', 'a')
file.writelines(Lines)
file.close()


# %%
file = open('1.txt', 'r')
Lines = file.readlines()
print(len(Lines))
file.close()
# %%
import matplotlib.pyplot as plt
import numpy as np
x = ['brutal force', 'simd', 'openmp', 'mpi']
y = [1911.45, 774.524, 290.699, 123.24]
fig = plt.figure()

plt.ylabel('processing time (ms)')
plt.xlabel('method')
plt.bar(x,y)
fig.savefig('result.jpg')

# %%
