#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from PIL import Image
from constants import  *
import pydicom as dicom
import torch.nn.functional as F
import numpy as np

class medicalImageDataset(torch.utils.data.Dataset):
    def __init__(self, flair_reg, mask_tumor, transformation = None):
        self.flair_reg = flair_reg
        self.mask_tumor = mask_tumor
        self.transformation = transformation

    def __len__(self):
        #return 1
        return len(self.mask_tumor)

    def __getitem__(self, index):
        transformed_channels = []
        if multiple_inputs:
            count = 0
            for i in range(len(self.flair_reg[index])):
                flair_path = self.flair_reg[index][i]
                mask_tumor_path = self.mask_tumor[index]
                flair = dicom.dcmread(flair_path).pixel_array
                mask_tumor = dicom.dcmread(mask_tumor_path).pixel_array
                X, y = self.transformation(Image.fromarray(flair, '1'),
                                           Image.fromarray(mask_tumor, '1')
                                           )
                transformed_channels.append(X)
                # if count < (len(self.flair_reg[index]) - 1):
                #     X, _ = self.transformation(Image.fromarray(flair, '1'),
                #                                  None
                #                                    )
                #     transformed_channels.append(X)
                # else:
                #     X, y = self.transformation(Image.fromarray(flair, '1'),
                #                                Image.fromarray(mask_tumor, '1')
                #                                )
                #     transformed_channels.append(X)
                #     return transformed_channels, y
            return transformed_channels, y

        else:
            flair_path = self.flair_reg[index]
            mask_tumor_path = self.mask_tumor[index]

            if self.transformation:
                flair = Image.open(flair_path)
                mask_tumor = Image.open(mask_tumor_path)
                X, y = self.transformation(flair, mask_tumor)


            return X, y, flair_path


