#!/usr/bin/env python
# coding: utf-8

# In[ ]:


learning_rate = 0.0001
train_batch_size = 10

train_directory = 'Braindata/TRAINING'
#train_directory = 'Braindata/TRAINING_old'
test_directory = 'Braindata/TEST'

multiple_inputs = False  ## if set to True, just take one image as an input else take 25
single_in_channels = 1
multiple_in_channels = 25
out_channels = 1
LOAD_MODEL = False
#BATCH_SIZE = 15
BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
WEIGHT_DECAY = 0
NUM_EPOCHS = 300
NUM_WORKERS = 0
PIN_MEMORY = False
SAVE_DIR = "../tumor/result/"

##### mask & flair resizing
width = 160
height = 240
# width = 320
# height = 260




