"""
Created on Fri Dec 23 12:27:22 2022

@author: Marco Penso
"""

import os
import numpy as np
import h5py
import cv2
import pydicom # for reading dicom files
import matplotlib.pyplot as plt
import shutil
from skimage import measure


drawing=False # true if mouse is pressed
mode=True

def paint_draw(event,former_x,former_y,flags,param):
    global current_former_x,current_former_y,drawing, mode
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        current_former_x,current_former_y=former_x,former_y
    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
                cv2.line(img,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),2)
                cv2.line(image_binary,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),2)
                current_former_x = former_x
                current_former_y = former_y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
            cv2.line(img,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),2)
            cv2.line(image_binary,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),2)
            current_former_x = former_x
            current_former_y = former_y
    return former_x,former_y
  
def imfill(img, dim):
    img = img[:,:,0]
    img = cv2.resize(img, (dim, dim))
    img[img>0]=255
    im_floodfill = img.copy()
    h, w = im_floodfill.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    return img | cv2.bitwise_not(im_floodfill)

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(x,",",y)
        X.append(y)
        Y.append(x)
        cv2.destroyAllWindows()

def convert_to_uint8(image):
    image = image - image.min()
    image = 255.0*np.divide(image.astype(np.float32),image.max())
    return image.astype(np.uint8)

def crop_or_pad_slice_to_size(slice, nx, ny):
    
    if len(slice.shape) == 3:
        stack = [slice[:,:,0], slice[:,:,1], slice[:,:,2]]
        RGB = []
    else:
        stack = [slice]
    
    for i in range(len(stack)):
        
        img = stack[i]
            
        x, y = img.shape
        
        x_s = (x - nx) // 2
        y_s = (y - ny) // 2
        x_c = (nx - x) // 2
        y_c = (ny - y) // 2
    
        if x > nx and y > ny:
            slice_cropped = img[x_s:x_s + nx, y_s:y_s + ny]
        else:
            slice_cropped = np.zeros((nx, ny))
            if x <= nx and y > ny:
                slice_cropped[x_c:x_c + x, :] = img[:, y_s:y_s + ny]
            elif x > nx and y <= ny:
                slice_cropped[:, y_c:y_c + y] = img[x_s:x_s + nx, :]
            else:
                slice_cropped[x_c:x_c + x, y_c:y_c + y] = img[:, :]
        if len(stack)>1:
            RGB.append(slice_cropped)
    
    if len(stack)>1:
        return np.dstack((RGB[0], RGB[1], RGB[2]))
    else:
        return slice_cropped
    
def crop_or_pad_slice_to_size_specific_point(slice, nx, ny, cx, cy):
    
    if len(slice.shape) == 3:
        stack = [slice[:,:,0], slice[:,:,1], slice[:,:,2]]
        RGB = []
    else:
        stack = [slice]
        
    for i in range(len(stack)):
        img = stack[i]
        x, y = img.shape
        y1 = (cy - (ny // 2))
        y2 = (cy + (ny // 2))
        x1 = (cx - (nx // 2))
        x2 = (cx + (nx // 2))
    
        if y1 < 0:
            img = np.append(np.zeros((x, abs(y1))), img, axis=1)
            x, y = img.shape
            y1 = 0
        if x1 < 0:
            img = np.append(np.zeros((abs(x1), y)), img, axis=0)
            x, y = img.shape
            x1 = 0
        if y2 > 512:
            img = np.append(img, np.zeros((x, y2 - 512)), axis=1)
            x, y = img.shape
        if x2 > 512:
            img = np.append(img, np.zeros((x2 - 512, y)), axis=0)
    
        slice_cropped = img[x1:x1 + nx, y1:y1 + ny]
        if len(stack)>1:
            RGB.append(slice_cropped)
        
    if len(stack)>1:
        return np.dstack((RGB[0], RGB[1], RGB[2]))
    else:
        return slice_cropped
    

input_path = r'F:/Myo_scar/esempio2'
crop = 150
path_pre = os.path.join(input_path, 'pre', 'LGE', 'raw')
path_post = os.path.join(input_path, 'post', 'LGE', 'raw')

for _ in range(2):
    path_pre = os.path.join(path_pre, os.listdir(path_pre)[0])
    path_post = os.path.join(path_post, os.listdir(path_post)[0])

if len(os.listdir(path_pre)) != len(os.listdir(path_post)):
    raise Exception('number of file in pre %s and post %s is not equal' % (
            len(os.listdir(path_pre)), len(os.listdir(path_post))))

datafile = {}
datafile['mask_pre']=[]
datafile['mask_post']=[]
datafile['img_pre']=[]
datafile['img_post']=[]
tit=['epicardium', 'endocardium','rv']
for n in range(len(os.listdir(path_pre))):
    
    for file in ['pre', 'post']:
        if file == 'pre':
            data_row_img = pydicom.dcmread(os.path.join(path_pre, os.listdir(path_pre)[n]))
        else:
            data_row_img = pydicom.dcmread(os.path.join(path_post, os.listdir(path_post)[n]))
        for ii in range(3):
            img = data_row_img.pixel_array
            img = crop_or_pad_slice_to_size(img, crop, crop)
            if ii == 0:
                if file == 'pre':
                    datafile['img_pre'].append(img)
                else:
                    datafile['img_post'].append(img)
            dim = img.shape[0]
            img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) 
            img = cv2.resize(img, (400, 400), interpolation = cv2.INTER_CUBIC)
    
            image_binary = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    
            cv2.namedWindow(tit[ii])
            cv2.setMouseCallback(tit[ii],paint_draw)
            while(1):
                cv2.imshow(tit[ii],img)
                k=cv2.waitKey(1)& 0xFF
                if k==27: #Escape KEY
                    if ii==0:
                        
                        im_out1 = imfill(image_binary, dim)
                        im_out1[im_out1>0]=255
                        
                    elif ii==1:
                                                
                        im_out2 = imfill(image_binary, dim)
                        im_out2[im_out2>0]=255
                    
                    elif ii==2:
                        im_out3 = imfill(image_binary, dim)
                        im_out3[im_out3>0]=255
                    break
            cv2.destroyAllWindows()
            
        im_out1[im_out1>0]=1
        im_out2[im_out2>0]=1
        im_out3[im_out3>0]=1
        #mask = im_out1 - im_out2
        mask = im_out1
        mask = mask + im_out3
        mask[mask>0]=1
        if file == 'pre':
            datafile['mask_pre'].append(mask)
        else:
            datafile['mask_post'].append(mask)


CX = []
CY = []
LEN_X = []
LEN_Y = []
for i in range(2):
    if i == 0:
        mask = datafile['mask_pre'][0]
        img = datafile['img_pre'][0]
    else:
        mask = datafile['mask_post'][0]
        img = datafile['img_post'][0]
    
    temp_img = mask.copy()
    temp_img[temp_img>0]=255
    temp_img = np.expand_dims(temp_img, -1)
    temp_img = imfill(temp_img, temp_img.shape[0])         
    contours, hier = cv2.findContours(temp_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    top_left_x = 1000
    top_left_y = 1000
    bottom_right_x = 0
    bottom_right_y = 0
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        if x < top_left_x:
            top_left_x = x
        if y < top_left_y:
            top_left_y= y
        if x+w-1 > bottom_right_x:
            bottom_right_x = x+w-1
        if y+h-1 > bottom_right_y:
            bottom_right_y = y+h-1        
    top_left = (top_left_x, top_left_y)
    bottom_right = (bottom_right_x, bottom_right_y)
    #print('top left=',top_left)
    #print('bottom right=',bottom_right)
    cx = int((top_left[1]+bottom_right[1])/2)   #row
    cy = int((top_left[0]+bottom_right[0])/2)   #column
    len_x = int(bottom_right[1]-top_left[1])
    len_y = int(bottom_right[0]-top_left[0])
    CX.append(cx)
    CY.append(cy)
    LEN_X.append(len_x)
    LEN_Y.append(len_y)
    
crop = 90
for ii in range(len(datafile['mask_pre'])):
    
    img_pre = datafile['img_pre'][ii].copy() # Image to be aligned.
    img_post = datafile['img_post'][ii].copy()  # Reference image.
    
    img_pre[datafile['mask_pre'][ii]==0] = 0
    img_post[datafile['mask_post'][ii]==0] = 0
    
    img_pre = crop_or_pad_slice_to_size_specific_point(img_pre, crop, crop, CX[0], CY[0])
    img_post = crop_or_pad_slice_to_size_specific_point(img_post, crop, crop, CX[-1], CY[-1])
    
    #Convert to grayscale.
    img_pre_8 = convert_to_uint8(img_pre)
    img_post_8 = convert_to_uint8(img_post)
    height, width = img_pre.shape
    
    #Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(nfeatures=500,edgeThreshold=0)
    
    #Find keypoints and descriptors.
    #The first arg is the image, second arg is the mask which is not required in this case.
    kp1, d1 = orb_detector.detectAndCompute(img_pre_8, None)
    kp2, d2 = orb_detector.detectAndCompute(img_post_8, None)
    
    #plot
    img1_kp = cv2.drawKeypoints(img_pre_8, kp1, None, color=(0,255,0), flags=cv2.DrawMatchesFlags_DEFAULT)
    img2_kp = cv2.drawKeypoints(img_post_8, kp2, None, color=(0,255,0), flags=cv2.DrawMatchesFlags_DEFAULT)

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.imshow(img1_kp)
    ax2 = fig.add_subplot(122)
    ax2.imshow(img2_kp)
    plt.show()
    
    #Match features between the two images.
    #We create a Brute Force matcher with Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    
    #Match the two sets of descriptors.
    matches = matcher.match(d1, d2)
    
    #Sort matches on the basis of their Hamming distance.
    matches.sort(key = lambda x: x.distance)
    
    #Take the top 90 % matches forward.
    matches = matches[:int(len(matches)*0.9)]
    no_of_matches = len(matches)
    
    #Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))
    
    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt
    
    #Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
    
    #Use this matrix to transform the colored image wrt the reference image.
    transformed_img = cv2.warpPerspective(img_pre, homography, (width, height))
        
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    plt.title('img pre')
    ax1.imshow(img_pre)
    ax2 = fig.add_subplot(132)
    plt.title('img post')
    ax2.imshow(img_post)
    ax3 = fig.add_subplot(133)
    plt.title('img pre transf')
    ax3.imshow(transformed_img)
    plt.show()


'''
color = [(255,0,0)]

labels_mask = measure.label(mask)                       
regions = measure.regionprops(labels_mask)
regions.sort(key=lambda x: x.area, reverse=True)
if len(regions) > 1:
    for rg in regions[1:]:
        labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
labels_mask[labels_mask!=0] = 1
mask = labels_mask.astype(np.uint8)
        
contours_mask, _ = cv2.findContours(image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

cv2.drawContours(image=img2, contours=contours_mask, contourIdx=-1, color=color[0], thickness=1, lineType=cv2.LINE_AA)
cv2.drawContours(image=transformed_img, contours=contours_mask, contourIdx=-1, color=color[0], thickness=1, lineType=cv2.LINE_AA)

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.imshow(img2)
ax2 = fig.add_subplot(122)
ax2.imshow(transformed_img)
plt.show()
'''
