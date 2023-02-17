#%% 
import numpy as np
import cv2
import glob
import pickle
import PySimpleGUI as sg
import os

# %%
def center_detect(calibration_path):
    calibrationImagePath = os.path.join(calibration_path, 'calibration.tif')
    calibrationImage = cv2.imread(calibrationImagePath, cv2.IMREAD_GRAYSCALE)                                
    kernel = np.array([[-1, -1,  -1],
                   [-1,  9, -1],
                    [-1, -1,  -1]])
    calibrationImage_sharp = cv2.filter2D(src = calibrationImage, ddepth = -1, kernel = kernel)
    img = 255 - calibrationImage_sharp 
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 0
    params.maxThreshold = 40
    # ! Filter by area
    params.filterByArea = True
    radius = round(25/2) # * calculated based on 250 um / 10 um
    params.minArea = 400
    # ! Filter by circularity
    params.filterByCircularity = True
    params.minCircularity = 0.8
    # ? Filter by Convexity, may be useful
    detector = cv2.SimpleBlobDetector_create(params)
    # ! detect blob
    keypoints = detector.detect(img)
    output_image = cv2.cvtColor(calibrationImage,cv2.COLOR_GRAY2RGB)
    for point in keypoints:
        x = int(point.pt[0])
        y = int(point.pt[1])
        cv2.circle(output_image, (x,y), 0, (0, 0, 255), 2)   
    output_image = cv2.drawKeypoints(output_image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    print('-------detected finished ----------') 
    return output_image, keypoints
    #cv2.imshow("circular detection",output_image)
    #while True:
    #    key = cv2.waitKey(1) & 0xFF
    #    if key == 27:
    #        break
    #cv2.destroyAllWindows()
# %%
'''
calibrationImagePath = '/home/liu/Desktop/MLA_DATA/13um_cali/calibration.tif'
calibrationImage = cv2.imread(calibrationImagePath, cv2.IMREAD_GRAYSCALE)
# ! Blur use kernel
#calibrationImage = cv2.blur(calibrationImage,(3,3))
# ! sharpen image with kernel
kernel = np.array([[-1, -1,  -1],
                   [-1,  9, -1],
                    [-1, -1,  -1]])
calibrationImage_sharp = cv2.filter2D(src = calibrationImage, ddepth = -1, kernel = kernel)




img = 255 - calibrationImage_sharp 
params = cv2.SimpleBlobDetector_Params()
# ! threshold defined 
params.minThreshold = 0
params.maxThreshold = 40
# ! Filter by area
params.filterByArea = True
radius = round(25/2) # * calculated based on 250 um / 10 um
params.minArea = 450
# ! Filter by circularity
params.filterByCircularity = True
params.minCircularity = 0.8
# ? Filter by Convexity, may be useful
detector = cv2.SimpleBlobDetector_create(params)
# ! detect blob
keypoints = detector.detect(img)
output_image = cv2.cvtColor(calibrationImage,cv2.COLOR_GRAY2RGB)
for point in keypoints:
    x = int(point.pt[0])
    y = int(point.pt[1])
    cv2.circle(output_image, (x,y), 0, (0, 0, 255), 2)   
output_image = cv2.drawKeypoints(output_image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
cv2.imshow("circular detection",output_image)
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
cv2.destroyAllWindows()
'''
# %%

# %%
