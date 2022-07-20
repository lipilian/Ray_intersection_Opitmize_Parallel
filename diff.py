
# %%
import numpy as np  

dataA = np.loadtxt('BrutalForceOutput.txt')
dataB = np.loadtxt('OpenMpOutput.txt')

diff_index = []
diff_value = []
true_value = []
nonZero = 0
for i in range(len(dataA)):
    if dataA[i] != 0:
        nonZero += 1
    
    diff = dataA[i] - dataB[i]
    if diff != 0:
        diff_value.append(diff)
        diff_index.append(i)
        true_value.append(dataA[i])
        #print('they are different for index: {}'.format(i))
print('Number of different voxels: {}'.format(len(diff_index)))
diff_value = np.array(diff_value)
#diff_value = np.abs(diff_value)
diff_sum = np.sum(diff_value)
print('Sum of different voxels: {}'.format(diff_sum))
print('Average of different voxels: {}'.format(diff_sum / len(diff_index)))
true_value = np.array(true_value)
true_value[true_value ==0] = 1
percent = np.divide(diff_value, np.array(true_value))
print('average of percantage: {}', np.mean(percent))
print('overall nonzero number: {}'.format(nonZero))

# %%
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
plt.hist(percent, bins =50, range=(0,1));
# %%

