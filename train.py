#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import pydicom as dicom
import torch.nn as nn
from dataloader import medicalDataset
import torch.optim as optim
import constants as const
from sklearn.metrics import confusion_matrix
from constants import *
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from model import Segmentation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from PIL import Image
print(device)
import cv2
from torch.utils.data import DataLoader
import random
import numpy as np
from sklearn.metrics import jaccard_score
from imageloader import medicalImageDataset
import matplotlib.image as mpimg
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img), t(mask)

        return img, mask

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    y_pred_f = torch.where(y_pred_f > 0.5, 1.0, 0.0)
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)



def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def tversky(y_true, y_pred, smooth=1, alpha=0.7):
    y_true_pos = torch.flatten(y_true)
    y_pred_pos = torch.flatten(y_pred)
    true_pos = torch.sum(y_true_pos * y_pred_pos)
    false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)




### save checkpoint

### load checkpoint
def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def load_data(path):
    X = []
    y = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if subdir.__contains__('MaskTumor') and file.endswith('.jpg'):
                y.append(os.path.join(subdir, file))
            if subdir.__contains__('FLAIRreg') and file.endswith('.jpg'):
                X.append(os.path.join(subdir, file))
        if len(X) != len(y):
            print(subdir)

    return X, y

def load_multiple(path):
    X = []
    y = []
    for subdir, dirs, files in os.walk(path):
        channels = []
        last_mask = None
        for file in files:
            if subdir.__contains__('MaskTumor') and file.endswith('.jpg'):
                last_mask = os.path.join(subdir, file)
            if subdir.__contains__('FLAIRreg') and file.endswith('.jpg'):
                channels.append(os.path.join(subdir, file))

        ## Flairreg
        if len(channels) > 0:
            X.append(channels)
        ### Masks
        if last_mask is not None:
            y.append(last_mask)


    return X, y

def load_single_channel():
    X_train, y_train = load_data(const.train_directory)
    X_test, y_test = load_data(const.test_directory)

    return X_train, y_train, X_test, y_test

def down_sample_val_data(x1, y1, x2, y2):
    #remove black images
    x1, y1 = down_sample_data(x1, y1)
    for i in range(int(len(x2)/10)):
        x1.append(x2[i])
        y1.append(y2[i])
    return x1, y1


def load_multiple_channel():
    X_train, y_train = load_multiple(const.train_directory)
    X_test, y_test = load_multiple(const.test_directory)

    return X_train, y_train, X_test, y_test


def down_sample_data(X_train, y_train):
    #print(len(X_train))

    transform = Compose([ transforms.ToTensor() ])
    missing_tumors = 0
    indices = []
    filtered_indices = []
    filtered_flair = []
    filtered_masks = []
    #random.shuffle(indices)
    for i in range(len(y_train)):
        mask_tumor = Image.open(y_train[i])
        _, y = transform(mask_tumor, mask_tumor)
        if torch.max(y) == 0 :
            missing_tumors += 1
            indices.append(i)
        else:
            filtered_flair.append(X_train[i])
            filtered_masks.append(y_train[i])

    print("Downsampling data")
    print(missing_tumors / len(y_train))
    ### remove 50% of the data
    for i in range(int(len(indices)//2)):
        X_train.pop(indices[i])
        y_train.pop(indices[i])
    return filtered_flair, filtered_masks


def get_loaders(batch_size, num_workers=0, pin_memory=False, ):
    transform = Compose([transforms.Resize((const.width, const.height)), transforms.ToTensor(), ])
    medical_Data = []
    if multiple_inputs:
        X_train, y_train, X_test, y_test = load_multiple_channel()
    else:
        X_train, y_train, X_test, y_test = load_single_channel()

    X_train, y_train = down_sample_data(X_train,y_train)
    X_test, y_test = down_sample_val_data(X_test, y_test, X_train, y_train)


    train_ds = medicalImageDataset(
        flair_reg = X_train,
        mask_tumor=y_train,
        transformation=transform,
    )


    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        #shuffle=True,
    )

    val_ds = medicalImageDataset(
        flair_reg=X_test,
        mask_tumor=y_test,
        transformation=transform,
    )


    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    js_score = 0

    predicted_array = []
    target_array = []

    with torch.no_grad():
        for x, y, z in loader:
            y = torch.reshape(y, (BATCH_SIZE, 1, const.width, const.height))

            if multiple_inputs:
                stack_x= x[0]
                for i in range(1, len(x)):
                    stack_x = torch.cat((x[i], stack_x), dim=1)
                if len(x) < multiple_in_channels:
                    missing = multiple_in_channels - len(x)
                    for i in range(missing):

                        duplicate_flair = random.randint(0, len(x) - 1)
                        stack_x = torch.cat((stack_x, x[duplicate_flair]), dim=1)

                predicted_image = torch.sigmoid(model(stack_x))
            else:
                x, y = x.to(device), y.to((device))
                predicted_image = torch.sigmoid(model(x))
                #print(jaccard_score(y, predicted_image))
                predicted = predicted_image.to(torch.int16)
                predicted = predicted.squeeze(0)
                predicted = predicted.squeeze(0)
                predicted = predicted.flatten()
                predicted =  predicted.cpu()
                predicted = predicted.detach().numpy()
                for i in range(const.width*const.height):
                    if predicted[i] > 0.5:
                        predicted[i] = 1
                    else:
                        predicted[i] = 0



                squeezed_y = y.to(torch.int16)
                squeezed_y = squeezed_y.squeeze(0)
                squeezed_y = squeezed_y.squeeze(0)
                #print('jaccard_score')
                #js_score += jaccard_score(squeezed_y.flatten(), predicted)
                #print(js_score)
                predicted_array.append(predicted)
                target_array.append(squeezed_y)


            preds = predicted_image

            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    accur = num_correct/num_pixels
    dice_overall = dice_score/len(loader)
    print(
        f"Got {num_correct}/{num_pixels} with acc {accur:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")

    # predicted_array = np.concatenate(predicted_array).flat
    # target_array = np.concatenate(target_array).flat
    # iou = jaccard_score(target_array, predicted_array)
    # print(iou)
    model.train()
    jicard_coeff = dice_overall/(2- dice_overall)

    return dice_overall, jicard_coeff, accur


def correct_pixel(img):
    for i in range(const.width):
        for j in range(const.height):
            if img[i][j] > 0.5:
                img[i][j] = 1
            else:
                img[i][j] = 0

    return img

#we just store target * predicted images
def output_images(flair, predicted_image, y, epoch, batch_idx):
    w = 20
    h = 20
    fig = plt.figure(figsize=(20, 20))
    columns = 3
    rows = 1
    img = torch.reshape(y, (const.width, const.height))
    img = img.cpu()
    img = img.detach().numpy()
    fig.add_subplot(rows, columns, 1)
    #flair_path = flair[0].replace("jpg", "dcm")
    ds  = mpimg.imread(flair[0])
    #ds = ds.pixel_array

    im = cv2.imread(flair[0])
    #print(flair[0])
    im_resized = cv2.resize(im, (const.height, const.width), interpolation=cv2.INTER_LINEAR)

    plt.imshow(cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB))
    #plt.imshow(ds, cmap=plt.cm.bone)
    for i in range(2, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        #plt.savefig('img.png')
        img = torch.reshape(predicted_image, (const.width, const.height))
        img = img.cpu()
        img = img.detach().numpy()
        img = correct_pixel(img)

    plt.savefig(const.SAVE_DIR+str(epoch)+'_'+str(batch_idx)+'_result.png')
    plt.close()

def find_val_loss(val_loader, model, loss_fn):
    loop = tqdm(val_loader, leave=True)
    mean_loss = None
    with torch.no_grad():
        for batch_idx, (x, y, z) in enumerate(loop):
            x, y = x.to(device), y.to((device))
            predicted_image = model(x)
            y = torch.where(y > 0.5, 1.0, 0.0)
            loss = loss_fn(predicted_image.float(), torch.reshape(y.float(), (BATCH_SIZE, out_channels,  const.width, const.height))) + dice_coef_loss(torch.reshape(y.float(), (BATCH_SIZE, out_channels,  const.width, const.height)), predicted_image.float())
            mean_loss = loss.item()
            # update progress bar
            loop.set_postfix(loss=loss.item())

    return mean_loss


def train_fn(train_loader, model, optimizer, loss_fn, epoch):
    loop = tqdm(train_loader, leave=True)
    mean_loss = None
    for batch_idx, (x, y, z) in enumerate(loop):

        stack_x = x[0]

        if multiple_inputs:
            for i in range(1, len(x)):
                stack_x = torch.cat((x[i], stack_x), dim=1)
            if len(x) < multiple_in_channels:
                missing = multiple_in_channels - len(x)
                for i in range(missing):

                    duplicate_flair = random.randint(0, len(x) - 1)
                    stack_x = torch.cat((stack_x, x[duplicate_flair]), dim=1)

            predicted_image = model(stack_x)
        else:
            x, y = x.to(device), y.to((device))
            #x = torch.squeeze(x, 1)
            #x = x.transpose(2, 3)
            predicted_image = model(x)

        #t1 = predicted_image.detach().numpy()
        #loss = loss_fn(torch.flatten(predicted_image), torch.flatten(y))

        y = torch.where(y > 0.5, 1.0, 0.0)

        loss = loss_fn(predicted_image.float(), torch.reshape(y.float(), (BATCH_SIZE, out_channels,  const.width, const.height))) + dice_coef_loss(torch.reshape(y.float(), (BATCH_SIZE, out_channels,  const.width, const.height)), predicted_image.float())
        print(loss)
        output_images(z, predicted_image, torch.reshape(y.float(), (1, 1,  const.width, const.height)), epoch, batch_idx)
        mean_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    return mean_loss

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return torch.pow((1 - tv), gamma)


def plot_combine_graphs(loss_values, val_loss_values, accuracies, val_accuracies, filename):
    length = len(loss_values)
    t_epochs = []

    for i in range(length):
        t_epochs.append(i)
    plt.plot(t_epochs, loss_values)
    plt.plot(t_epochs, val_loss_values)

    plt.plot(t_epochs, accuracies)
    plt.plot(t_epochs, val_accuracies)

    ylable = filename.replace('.png', '')
    plt.ylabel(ylable)
    plt.xlabel('epoch')
    plt.legend(["train_loss", "test_loss", "train_accuracy",  "test_accuracy"])
    plt.savefig(filename)
    plt.close()

def plot_graphs(train_arr, test_arr, filename):
    length = len(train_arr)
    t_epochs = []

    for i in range(length):
        t_epochs.append(i)
    plt.plot(t_epochs, train_arr)
    plt.plot(t_epochs, test_arr)
    ylable = filename.replace('.png', '')
    plt.ylabel(ylable)
    plt.xlabel('epoch')
    plt.legend(["train"+ylable, "test"+ylable])
    plt.savefig(filename)
    plt.close()


def main():
    ## define model
    if multiple_inputs:
        model = Segmentation(in_channels=multiple_in_channels, out_channels=1).to(device)
    else:
        model = Segmentation(in_channels=single_in_channels, out_channels=1).to(device)

    ## define loss
    loss_fn = nn.BCEWithLogitsLoss()
    # = nn.BCEWithLogitsLoss(pos_weigth= )


    ## define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # load train & test dataset ->val_loader means test dataset
    train_loader, val_loader = get_loaders(
        BATCH_SIZE,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    ## we can resume training from last checkppoint by setting LOAD_MODEL= True & updating filename herr
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint39.pth.tar"), model, optimizer)

    loss_values = []
    val_loss_values = []
    dice_scores = []
    jaccard_scores = []
    accuracies = []


    val_dice_scores = []
    val_jaccard_scores = []
    val_accuracies = []
    model.train()
    for epoch in range(NUM_EPOCHS):
        loss = train_fn(train_loader, model, optimizer, loss_fn, epoch)
        loss_values.append(loss)
        test_model = model
        val_loss = find_val_loss(val_loader, test_model, loss_fn)
        val_loss_values.append(val_loss)

        if epoch > 60:
            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, "my_checkpoint" + str(epoch) + ".pth.tar")

        # check train accuracy
        dice_s , js_s, accuracy = check_accuracy(train_loader, model, device=device)
        dice_scores.append(dice_s)
        jaccard_scores.append(js_s)
        accuracies.append(float(accuracy))
        #plot_graphs(loss_values, 'train_total_loss.png')




        # plot_graphs(dice_scores, 'train_dice_scores.png')
        # plot_graphs(jaccard_scores, 'train_jaccard_score.png')
        # plot_graphs(accuracies, 'train_accuracy.png')

        # check val accuracy

        test_dice_s, test_js_s, test_accuracy = check_accuracy(val_loader, model, device=device)
        val_dice_scores.append(test_dice_s)
        val_jaccard_scores.append(test_js_s)
        val_accuracies.append(test_accuracy)

        plot_graphs(dice_scores, val_dice_scores, 'dice_scores.png')
        plot_graphs(jaccard_scores, val_jaccard_scores, 'jaccard_score.png')
        plot_graphs(loss_values, val_loss_values, 'loss.png')
        plot_graphs(accuracies, val_accuracies, 'accuracy.png')
        
        #plot_combine_graphs(loss_values, val_loss_values, accuracies, val_accuracies, 'accuracy.png')




    plt.plot(loss_values)
    plt.show()



if __name__ == "__main__":
    main()

