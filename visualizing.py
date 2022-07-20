# %% 
import os
import io
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# %% prerun to get datainformation from user
RayCounter_BrutalForce= np.loadtxt('BrutalForceOutput.txt')
RayCounter_CUDA = np.loadtxt('CudaOutput.txt')
NumZ = 401
NumX = 40
NumY = 120
Xmin = -1
Xmax = 4.0
Ymin = -1
Ymax = 14
Zmin = -40
Zmax = 40
deltaX = (Xmax - Xmin)/NumX
deltaY = (Ymax - Ymin)/NumY
deltaZ = 80/NumZ
X = np.zeros(len(RayCounter_BrutalForce))
Y = np.zeros(len(RayCounter_BrutalForce))
Z = np.zeros(len(RayCounter_BrutalForce))
for i in range(len(RayCounter_BrutalForce)):
    Zvalue = int(i/(NumX*NumY)) * deltaZ + Zmin
    mod = i % (NumX*NumY)
    Xvalue = int(mod/NumY) * deltaX + Xmin
    mod = mod % NumY
    Yvalue = mod * deltaY + Ymin
    X[i] = Xvalue
    Y[i] = Yvalue
    Z[i] = Zvalue

# %%
%matplotlib
threshold = 1000
X1 = X[RayCounter_BrutalForce > threshold]
Y1 = Y[RayCounter_BrutalForce > threshold]
Z1 = Z[RayCounter_BrutalForce > threshold]
RayCounter_BrutalForce = RayCounter_BrutalForce[RayCounter_BrutalForce > threshold]
fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(X1, Y1, Z1, c = RayCounter_BrutalForce, cmap = 'jet', vmin = threshold, vmax = np.max(RayCounter_BrutalForce), s= 1)

ax.view_init(15.3284, -2.6613)

ax.set_xlim(-1, 4)
ax.set_ylim(-1,14)
ax.set_zlim(-40,40)
#%%fig.savefig(BrutalForce'  + '.jpg', dpi = 300)
threshold = 1000
X2 = X[RayCounter_CUDA > threshold]
Y2 = Y[RayCounter_CUDA > threshold]
Z2 = Z[RayCounter_CUDA > threshold]
RayCounter_CUDA = RayCounter_CUDA[RayCounter_CUDA > threshold]
fig = plt.figure(2)
plt.clf()
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(X2, Y2, Z2, c = RayCounter_CUDA, cmap = 'jet', vmin = threshold, vmax = np.max(RayCounter_CUDA), s= 1)

ax.view_init(15.3284, -2.6613)

ax.set_xlim(-1, 4)
ax.set_ylim(-1,14)
ax.set_zlim(-40,40)

# %%
