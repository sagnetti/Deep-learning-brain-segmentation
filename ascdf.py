#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pydicom as dicom
import matplotlib.pylab as plt
import numpy as np
import cv2

# specify your image path
image_path = 'test.dcm'
ds = dicom.dcmread(image_path)

pixel_array_numpy = ds.pixel_array

image_format = '.jpg' # or '.png'

image_path = image_path.replace('.dcm', image_format)
#cv2.imshow("img", pixel_array_numpy)
#cv2.imwrite(image_path, pixel_array_numpy)
ds = dicom.dcmread("test.dcm")
ds = ds.pixel_array
#ds = np.resize(ds,(240,160))
#cv2.imshow("img",ds)
ds[ds > 0] = 255

#ds = np.resize(ds,(240,160))

cv2.imwrite(image_path, ds)
plt.imshow(ds, cmap=plt.cm.bone)
plt.savefig('_result.png')
plt.close()





