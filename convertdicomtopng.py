# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 11:25:12 2020

@author: chenj
"""

import pydicom

import numpy as np




import cv2


import os
import glob

dicompath='D:/Dataset/RIDER/RIDER Lung CT/RIDER-1129164940/09-20-2006-1-96508/4-24533/'

jpgpath='D:/Dataset/RIDER/RIDER Lung CT/jpgfile/'

if not os.path.exists(jpgpath):
    os.mkdir(jpgpath)

Dicomfiles=glob.glob(dicompath+'/*.dcm')

for j in range(len(Dicomfiles)):

    ds = pydicom.dcmread(Dicomfiles[j])

    shape = ds.pixel_array.shape

# Convert to float to avoid overflow or underflow losses.
    image_2d = ds.pixel_array.astype(float)

# Value Range normalization    
    image_2d[image_2d>2000]=2000
    image_2d[image_2d<-2000]=-2000

# Rescaling grey scale between 0-255
    image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0

# Convert to uint
    image_2d_scaled = np.uint8(image_2d_scaled)
    image_2d_scaled = np.reshape(image_2d_scaled,(512,512,1))
    image_2d_scaled =np.concatenate([image_2d_scaled,image_2d_scaled],1)
    image_2d_scaled_rgb=np.concatenate([image_2d_scaled,image_2d_scaled,image_2d_scaled],2)


#save images    
    jpg_file=jpgpath+'/'+str(j).rjust(4,'0')+'.jpg'
    cv2.imwrite(jpg_file, image_2d_scaled_rgb)

