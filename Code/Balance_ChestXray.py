'''
## Informazioni sul dataset ##
Il dataset contiene 5863 immagini di radiografie di toraci di bambini suddivise in 2 classi:
 - Normal
  -Pneumonia (o Polmonite): ogni infiammazione, acuta o cronica del parenchima polmonare, 
                            causata principalmente da agenti patogeni batterici o virali.
                            
                            
MOONEY, P. Chest X-Ray (Pneumonia) Dataset (2018). Available from:
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia.

'''

import os
import random
import cv2
import glob
import numpy as np
import pandas as pd 
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.image import imread

import torch

from torchvision import transforms 
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler

from sklearn.model_selection import train_test_split

def main():
    path = '../Dataset/chest_xray'
    classes = os.listdir(path+'/chest_xray')
    print("\nNumero classi: ", len(classes))
    print("Classi: ", classes)
    
    
    ## 1. Check for DATA DISTRUBUTION  ##
    '''
    Data distribution: there is a strong imbalance in the data 
    (there are many times more X-rays with pneumonia than normal ones)
    '''
    os_normal = [os.path.join(path+'/chest_xray/NORMAL', filename) 
                     for filename in os.listdir(path+'/chest_xray/NORMAL')]
    os_pneumonia = [os.path.join(path+'/chest_xray/PNEUMONIA', filename) 
                       for filename in os.listdir(path+'/chest_xray/PNEUMONIA')]
    
    normal_len = len(os_normal)
    pneumonia_len = len(os_pneumonia)
    
    data_samplesize = pd.DataFrame.from_dict( {'Normal': [normal_len], 'Pneumonia': [pneumonia_len]})
    sns.barplot(data=data_samplesize).set_title('Dataset Inbalance', fontsize=20)
    plt.show()
    print("\nUnbalance:\nNormal: ", normal_len)
    print("Pneumonia: ", pneumonia_len)

    rand_samples = random.sample(os_normal, 5) + \
                   random.sample(os_pneumonia, 5)
    # Let's display X-rays with diagnoses
    plot_samples(rand_samples)
    plt.suptitle('Dataset Samples', fontsize=30)
    plt.show()
    
    
    ## 2. Preparing TRAIN, VALIDATION and TEST LOADER ##
    '''
    Let's duplicate the size of the dataset and split it with the following distrubution [Hold-Out method]:
        - Training: 70%
        - Validation: 20%
        - Testing: 10%
    '''
    norm_mean, norm_std = compute_img_mean_std(path + '/all images/')
    
    augment_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=1.0),
                                           transforms.RandomRotation(20),
                                           transforms.RandomGrayscale(),
                                           transforms.RandomAffine(translate=(0.05,0.05), degrees=0),
                                           transforms.Resize(256),
                                           transforms.CenterCrop(256),
                                           # Inception v3
                                           #transforms.Resize(299),
                                           #transforms.CenterCrop(299),
                                           transforms.ToTensor(),
                                           transforms.Normalize(norm_mean, norm_std)
                                           ])

    ds_transform = transforms.Compose([transforms.Resize(256),                        
                                      transforms.CenterCrop(256), 
                                      #InceptionV3
                                      #transforms.Resize(299),
                                      #transforms.CenterCrop(299),
                                      transforms.ToTensor(),
                                      transforms.Normalize(norm_mean, norm_std)
                                      ])

    dataset_clean = ImageFolder(path + '/chest_xray', transform=ds_transform)
    dataset_augment = ImageFolder(path + '/chest_xray', transform=augment_transform)
    dataset = ConcatDataset([dataset_clean, dataset_augment])

    print('\nLen dataset: ', len(dataset))
    train_ds, val_test_ds = train_test_split(dataset, train_size= 0.7, shuffle=True)
    print('Len Training dataset: ', len(train_ds))
    val_ds, test_ds = train_test_split(val_test_ds, test_size= 1/3, shuffle=True)
    print('Len Validation dataset: ', len(val_ds))
    print('Len Testing dataset: ', len(test_ds))
    
    datasets = [train_ds, val_ds, test_ds] 
    all_len = calculate_len_datasets(datasets)
    train_dl, val_dl, test_dl = get_loaders(datasets, all_len)
    
    show_batch(train_dl)


    # Saving Dataloaders
    torch.save(train_dl, 'ChestX_Ray_train_dl.pt')
    torch.save(val_dl, 'ChestX_Ray_val_dl.pt')
    torch.save(test_dl, 'ChestX_Ray_test_dl.pt')


    
# Function for plotting samples
def plot_samples(samples):  
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(30,8))
    for i in range(len(samples)):
        image = cv2.cvtColor(imread(samples[i]), cv2.COLOR_BGR2RGB)
        ax[i//5][i%5].imshow(image)
        if i<5:
            ax[i//5][i%5].set_title("Normal", fontsize=20)
        else:
            ax[i//5][i%5].set_title("Pneumonia", fontsize=20)
        ax[i//5][i%5].axis('off')


# Compute mean and std for normalization
def compute_img_mean_std(image_paths):
    """
        computing the mean and std of three channel on the whole dataset,
        first we should normalize the image from 0-255 to 0-1
    """

    ext = ['png', 'jpg', 'gif', 'jpeg']    # Add image formats here
    files = []
    [files.extend(glob.glob(image_paths + '*.' + e)) for e in ext]

    images = [cv2.imread(file) for file in files]
    img_h, img_w = 224, 224
    #img_h, img_w = 299, 299
    imgs = []
    means, stdevs = [], []

    for img in images:
        img = cv2.resize(img, (img_h, img_w))
        imgs.append(img)

    imgs = np.stack(imgs, axis=3)
    #print(imgs.shape)

    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # resize to one row
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()  # BGR --> RGB
    stdevs.reverse()

    print("\nnormMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    return means,stdevs


def calculate_len_datasets(datasets):
    '''
    Parameters
    ----------
    datasets : 
        Array: [train_dataset, val_dataset, test_dataset]
        
    Returns
    -------
    all_len: 
        Array: [len_train_ds, len_normal_train_ds, len_pneumonia_train_ds,
                len_val_ds, len_normal_val_ds, len_pneumonia_val_ds,
                len_test_ds, len_normal_test_ds, len_pneumonia_test_ds]
    '''
    all_len = []
    for ds in datasets:
        normal = 0
        pneumonia = 0
        for idx, (data, label) in enumerate(ds):
            if (label==0):
                normal = normal+1
            else:
                pneumonia = pneumonia+1
        tot = len(ds)
        print('\nTrain tot: ', tot)
        print('Train normal: ', normal)
        print('Train pneumonia: ', pneumonia)
        all_len.append(tot)
        all_len.append(normal)
        all_len.append(pneumonia)
    return all_len

def get_loaders(datasets, all_len):
    '''
    ## Create a loader for each dataset in input() ##
    Parameters
    ----------
    datasets : 
        Array: [train_dataset, val_dataset, test_dataset]
    all_len : 
        Array: [len_train_ds, len_normal_train_ds, len_pneumonia_train_ds,
                len_val_ds, len_normal_val_ds, len_pneumonia_val_ds,
                len_test_ds, len_normal_test_ds, len_pneumonia_test_ds]
        
    Returns
    -------
    loaders :
        Array: [train_loader, val_loader, test_loader]
    '''
    loaders = []
    x = 0
    for dataset in datasets:
        class_weights = [1.0/all_len[x+1], 1.0/all_len[x+2]]
        print('\nClass weights: ', class_weights)
        sample_weights = [0]*all_len[x]
        
        for idx, (data, label) in enumerate(dataset):
            class_weight = class_weights[label]
            sample_weights[idx] = class_weight
            
        sampler = WeightedRandomSampler(sample_weights, num_samples = all_len[x], replacement = True)
        loader = DataLoader(dataset, batch_size=128, sampler = sampler)
        loaders.append(loader)
        x = x + 3
    return loaders


# Let's see what the data with augmentation looks like
def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images[:60], nrow=8).permute(1, 2, 0))
        break
   
    
if __name__ == "__main__":
    main()
