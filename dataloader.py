#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from PIL import Image
from constants import  *
import pydicom as dicom
import torch.nn.functional as F
import numpy as np

### medical_dataset is an array [[index1, index2], [label1, label2]]
        ### medical_dataset is an array [[[index1], [label]], [input2, label2]]
        ### [[1dcm,2.dcm],[1dcm,2.dcm]], [1,2,3]]

class medicalDataset(torch.utils.data.Dataset):
    def __init__(self, flair_reg, mask_tumor, transformation = None):
        self.flair_reg = flair_reg
        self.mask_tumor = mask_tumor
        self.transformation = transformation

    def __len__(self):

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
            flair = dicom.dcmread(flair_path).pixel_array
            mask_tumor = dicom.dcmread(mask_tumor_path).pixel_array
            mask_tumor[mask_tumor > 0] = 255
            X, y = self.transformation(Image.fromarray(flair, '1'),
                                       Image.fromarray(mask_tumor, '1')
                                       )

            flair = np.int8(flair)
            X = torch.from_numpy(flair)
            mask_tumor = np.int8(mask_tumor)
            y = torch.from_numpy(mask_tumor)



            # transformed_img = torch.from_numpy(dicom.dcmread(self.flair_reg[index]).pixel_array)
            # X= F.interpolate(transformed_img, (width, height))

            X = X.type(torch.float)
            y = y.type(torch.LongTensor)
            # X.resize_(width, height)
            # y.resize_(width, height)
            return torch.unsqueeze(X, 0), torch.unsqueeze(y, 0)


