#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pydicom as dicom
import matplotlib.pylab as plt
import numpy as np
import cv2
rootdir = 'Braindata/'

X = []
Y = []
image_format = '.jpg'

for subdir, dirs, files in os.walk(rootdir):
    channels = []
    last_mask = None
    for file in files:
        if subdir.__contains__('FLAIRreg') and file.endswith('.dcm'):
            last_mask = file
            ds = dicom.dcmread( os.path.join(subdir, file))
            ds = ds.pixel_array
            #ds[ds > 0] = 255
            image_path = os.path.join(subdir, file).replace('.dcm', image_format)
            cv2.imwrite(image_path, ds)

        if subdir.__contains__('MaskTumor') and file.endswith('.dcm'):
            last_mask = file
            ds = dicom.dcmread(os.path.join(subdir, file))
            ds = ds.pixel_array
            ds[ds > 0] = 255
            image_path = os.path.join(subdir, file).replace('.dcm', image_format)
            cv2.imwrite(image_path, ds)

