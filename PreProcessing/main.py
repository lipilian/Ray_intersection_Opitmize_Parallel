# %%
import prerun
import importlib
importlib.reload(prerun)
import cv2
import numpy as np
import random
import time
import glob
import os
import matplotlib.image as mpimg
from skimage import morphology
import matplotlib.pyplot as plt
# %% calibration
OpticInfo = prerun.prerun() # * Optic Information
import Calibration; importlib.reload(Calibration)
detected_Image, keypoints = Calibration.center_detect(OpticInfo['calibration_path'])
# %% # * visualize the center detection image
cv2.imshow("circular detection",detected_Image)
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
cv2.destroyAllWindows()
# %% # * extract keypoints 
centers = []
for point in keypoints:
    x = point.pt[0]
    y = point.pt[1]
    centers.append([x,y])
centers = np.array(centers)
rounded_centers = np.round(centers).astype(int)
diameter = OpticInfo['MLA_size_mm'] / OpticInfo['pixel_mm']
assert diameter % 2 == 1
radius = (diameter - 1)/2
x = np.linspace(-1 * radius,radius,num = int(diameter)).astype(int)
y = x
x, y = np.meshgrid(x,y)
x = x.flatten()
y = y.flatten()
Rays = []
for i, center in enumerate(centers):
    center_x = rounded_centers[i, 0]
    center_y = rounded_centers[i, 1]
    center_actual_x = center[0]
    center_actual_y = center[1]
    pixel_x_lists = center_x + x
    pixel_y_lists = center_y + y    
    for pixel_x,pixel_y in zip(pixel_x_lists, pixel_y_lists):
        Rays.append([pixel_x, pixel_y, center_actual_x, center_actual_y])
Rays = np.array(Rays)
# ? randomly check several points if it is correct 
calibrationImagePath = os.path.join(OpticInfo['calibration_path'], 'calibration.tif')
calibrationImage = cv2.imread(calibrationImagePath, cv2.IMREAD_GRAYSCALE)
calibrationImage = cv2.cvtColor(calibrationImage,cv2.COLOR_GRAY2RGB)
Num_pixels = diameter**2
Selected_centers_index = random.sample(range(0, len(centers)), 10) # * select 10 random points to confirm center and affiliated points region
for i in Selected_centers_index:
    start_index = int(i * Num_pixels)
    end_index = int((i + 1) * Num_pixels)
    x_min = int(np.min(Rays[start_index:end_index,0]))
    x_max = int(np.max(Rays[start_index:end_index,0]))
    y_min = int(np.min(Rays[start_index:end_index,1]))
    y_max = int(np.max(Rays[start_index:end_index,1]))
    x_center = rounded_centers[i,0]
    y_center = rounded_centers[i,1]
    cv2.rectangle(calibrationImage, (x_min, y_min), (x_max, y_max), (255,0,0), 2)
    cv2.circle(calibrationImage, (x_center, y_center), 0, (0,0,255), 2)

cv2.imshow("random check",calibrationImage)
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
cv2.destroyAllWindows()   
      
# %% # brutal force 
'''
threshold = OpticInfo['minIntensity']
ImagePath = OpticInfo['target_path']
ImageLists = glob.glob(os.path.join(ImagePath,"*.tif"))
img = mpimg.imread(ImageLists[0])
start_time = time.time()
img_binary = img < threshold
img_binary = morphology.area_closing(img_binary, 2)
Ifshow = False
if Ifshow:
    img_show = (img_binary * 255).astype('uint8')
    cv2.imshow("target img",img_show)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    cv2.destroyAllWindows()

Xmax = OpticInfo['xmax_mm'] # mm
Xmin = OpticInfo['xmin_mm'] # mm
Ymin = OpticInfo['ymin_mm'] # mm
Ymax = OpticInfo['ymax_mm'] # mm
Zmin = OpticInfo['dmin_mm'] # mm
Zmax = OpticInfo['dmax_mm'] # mm
MLA_diameter = OpticInfo['MLA_size_mm'] # mm
Pixel_size = OpticInfo['pixel_mm'] # mm
MLA_FLength = OpticInfo['MLA_F_mm'] # mm
deltaX = MLA_diameter # select dx
deltaY = MLA_diameter # select dy
NumX = int((Xmax - Xmin) / deltaX)
NumY = int((Ymax - Ymin) / deltaY)
NumZ = int(OpticInfo['dnum']  + 1)

RayCounts = np.zeros((NumX,NumY,NumZ), dtype = np.uint)
m, n = img_binary.shape
Z_level = np.linspace(Zmin, Zmax, NumZ)
start_time = time.time()
for Ray in Rays:
    x_position = int(Ray[0])
    y_position = int(Ray[1])
    if not img_binary[y_position, x_position]:
        for Zindex, Z in enumerate(Z_level):
            scale = Z / MLA_FLength
            newX, newY = (Ray[0:2] - Ray[2:4]) * Pixel_size * scale + Ray[2:4] * Pixel_size
            Xindex = round((newX - Xmin) / deltaX)
            Yindex = round((newY - Ymin) / deltaY)
            RayCounts[Xindex,Yindex,Zindex] += 1
end_time = time.time()

print('the runing time for brutal force is {} ms'.format((end_time-start_time) * 1000))'''
# %% 
'''
plt.figure()
plt.hist(RayCounts.flatten(), bins = 30, range = (20, np.max(RayCounts.flatten())));
plt.yscale('log')'''

# %% cuda optimization 
from numba import jit
from numba import cuda, types
import math
threshold = OpticInfo['minIntensity']
ImagePath = OpticInfo['target_path']
ImageLists = glob.glob(os.path.join(ImagePath,"*.tif"))
Xmax = OpticInfo['xmax_mm'] # mm
Xmin = OpticInfo['xmin_mm'] # mm
Ymin = OpticInfo['ymin_mm'] # mm
Ymax = OpticInfo['ymax_mm'] # mm
Zmin = OpticInfo['dmin_mm'] # mm
Zmax = OpticInfo['dmax_mm'] # mm
MLA_diameter = OpticInfo['MLA_size_mm'] # mm
Pixel_size = OpticInfo['pixel_mm'] # mm
MLA_FLength = OpticInfo['MLA_F_mm'] # mm
deltaX = MLA_diameter # select dx
deltaY = MLA_diameter # select dy
NumX = int((Xmax - Xmin) / deltaX)
NumY = int((Ymax - Ymin) / deltaY)
NumZ = int(OpticInfo['dnum']  + 1)
Z_level = np.linspace(Zmin, Zmax, NumZ)

# * grid and block dimension definition
Rays_Num = len(Rays)
DimBlock = 1024
DimGrid = math.ceil(Rays_Num/DimBlock)
# * image loading
img = mpimg.imread(ImageLists[200])
Num_img_per_loop = 100
img_stacks = np.empty([img.shape[0], img.shape[1], Num_img_per_loop], dtype = img.dtype)
for i in range(Num_img_per_loop):
    img_stacks[:,:,i] = mpimg.imread(ImageLists[i])
# %%
# * CUDA function ----- device code
@cuda.jit # (device = True)
def UpdateRays_CUDA(device_img, device_Rays, device_RayCounts, Z_level, threshold, MLA_FLength, Pixel_size, Xmin, Ymin, deltaX, deltaY, Num_images):
    # ! create constant memory arrays
    Constant_memory_Z_level = cuda.const.array_like(Z_level)
    gid = cuda.grid(1)    
    if gid < Constant_memory_Z_level.shape[0]:
        Z_level[gid] = Constant_memory_Z_level[gid]
    cuda.syncthreads()
    # ! start processing image stacks 
    
    if gid < device_Rays.shape[0]:
        x = int(device_Rays[gid, 0])
        y = int(device_Rays[gid, 1])
        for i in range(Num_images):
            if device_img[y, x, i] > threshold:
                for Zindex, Z in enumerate(Constant_memory_Z_level):
                    scale = Z / MLA_FLength
                    newX = (device_Rays[gid,0] - device_Rays[gid,2] ) * Pixel_size * scale + device_Rays[gid, 2] * Pixel_size
                    newY = (device_Rays[gid,1] - device_Rays[gid,3] ) * Pixel_size * scale + device_Rays[gid, 3] * Pixel_size
                    Xindex = round((newX - Xmin) / deltaX)
                    Yindex = round((newY - Ymin) / deltaY)
                    cuda.atomic.add(device_RayCounts, (Xindex,Yindex,Zindex, i), 1)  

# ! start kernel 
start_time = time.time()
device_img = cuda.to_device(img_stacks) # transfer image data to device
device_Rays = cuda.to_device(Rays) # transfer Rays data to device
device_Z_level = cuda.to_device(Z_level) # ! maybe not necessary
# create output array
RayCounts_CPU = np.zeros((NumX,NumY,NumZ, Num_img_per_loop), dtype = np.uint32)
device_RayCounts = cuda.to_device(RayCounts_CPU)


UpdateRays_CUDA[DimGrid, DimBlock](device_img, device_Rays, device_RayCounts, device_Z_level, threshold, MLA_FLength, Pixel_size, Xmin, Ymin, deltaX, deltaY, Num_img_per_loop)

RayCounts = device_RayCounts.copy_to_host()
end_time = time.time()
print('the runing time for  cuda optimization for each frame is {} ms'.format((end_time-start_time) * 1000 / Num_img_per_loop))
device = cuda.select_device(0)
device.reset()

# %%function to loop through entire image_stacks
def process_image_stacks(ImageLists, start_index, end_index):
    NumImages_unprocessed = len(ImageLists)
    img = mpimg.imread(ImageLists[0])
    img_stacks = np.empty([img.shape[0], img.shape[1], end_index - start_index], dtype = img.dtype)
    device_img = cuda.to_device(img_stacks)
    device_Rays = cuda.to_device(Rays)
    device_Z_level = cuda.to_device(Z_level)
    RayCounts_CPU = np.zeros((NumX,NumY,NumZ, end_index - start_index), dtype = np.uint32)
    device_RayCounts = cuda.to_device(RayCounts_CPU)
    UpdateRays_CUDA[DimGrid, DimBlock](device_img, device_Rays, device_RayCounts, device_Z_level, threshold, MLA_FLength, Pixel_size, Xmin, Ymin, deltaX, deltaY, Num_img_per_loop)
    RayCounts = device_RayCounts.copy_to_host()
    device = cuda.select_device(0)
    device.reset()
    return RayCounts
RayCounts = process_image_stacks(ImageLists,0, 50)

# %% 
plt.figure()
plt.hist(RayCounts[:,:,:,0].flatten(), bins = 30, range = (20, np.max(RayCounts.flatten())));
plt.yscale('log')
# %% visualize the cloud points
maxIntensity = 100
frame = 0
X_1D = np.linspace(Xmin + deltaX / 2, Xmax- deltaX/2, NumX)
Y_1D = np.linspace(Ymin + deltaY / 2, Ymax- deltaY/2, NumY)
Z_1D = Z_level
X_3D, Y_3D, Z_3D = np.meshgrid(X_1D, Y_1D, Z_1D, indexing = 'ij')
RayCounts_singleFrame = RayCounts[:,:,:,frame]
RayCounts_valid = RayCounts_singleFrame[RayCounts_singleFrame  >= maxIntensity]
X_3D_valid = X_3D[RayCounts_singleFrame  >= maxIntensity]
Y_3D_valid = Y_3D[RayCounts_singleFrame  >= maxIntensity]
Z_3D_valid = Z_3D[RayCounts_singleFrame  >= maxIntensity]

%matplotlib
fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(X_3D_valid, Y_3D_valid, Z_3D_valid, c = RayCounts_valid, cmap = 'jet', vmin = maxIntensity, vmax = np.max(RayCounts_valid), s= 1)
ax.set_xlim(Xmin, Xmax)
ax.set_ylim(Ymin, Ymax)
ax.set_zlim(Zmin, Zmax)
ax.view_init(37, 28)
        
        

# %%
