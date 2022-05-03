'''
## Informazioni sul dataset ##
Il dataset contiene 7022 immagini di risonanze magnetiche cerebrali umane suddivise in 4 classi:
 - Glioma: un tumore che si sviluppa a partire dalle cellule della glia (o cellule gliali) del sistema nervoso centrale
 - Meningioma: un tumore cerebrale che origina dalle meningi, ovvero le membrane protettive che circondano e proteggono 
               il cervello e il midollo spinale.
 - No tumor
 - Pituitary: un adenoma ipofisario, anche conosciuto come adenoma pituitario, è un tumore generalmente benigno 
              che colpisce l’ipofisi, una ghiandola dalle dimensioni molto ridotte che si trova sotto l’ipotalamo


Nickparvar, M. Brain Tumor MRI Dataset (2020). Available from: https://
www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset.

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
    path = '../Dataset/brain-tumor-mri-dataset'
    classes = os.listdir(path+'/brain-tumor-mri-dataset')
    print("\nNumero classi: ", len(classes))
    print("Classi: ", classes)
    
    
    ## 1. Check for DATA DISTRUBUTION  ##
    
    os_glioma = [os.path.join(path +'/brain-tumor-mri-dataset/glioma', filename) 
                 for filename in os.listdir(path +'/brain-tumor-mri-dataset/glioma')]
    os_meningioma = [os.path.join(path+'/brain-tumor-mri-dataset/meningioma', filename) 
                     for filename in os.listdir(path +'/brain-tumor-mri-dataset/meningioma')]
    os_notumor = [os.path.join(path +'/brain-tumor-mri-dataset/notumor', filename) 
                  for filename in os.listdir(path +'/brain-tumor-mri-dataset/notumor')]
    os_pituitary = [os.path.join(path +'/brain-tumor-mri-dataset/pituitary', filename) 
                    for filename in os.listdir(path +'/brain-tumor-mri-dataset/pituitary')]
    
    len_glioma = len(os_glioma)
    len_meningioma = len(os_meningioma)
    len_notumor = len(os_notumor)
    len_pituitary = len(os_pituitary)
    
    data_samplesize = pd.DataFrame.from_dict({'glioma': [len_glioma], 'meningioma': [len_meningioma],
                                              'notumor': [len_notumor], 'pituitary': [len_pituitary]})
    sns.barplot(data=data_samplesize).set_title('Dataset Inbalance', fontsize=20)
    plt.show()
    print('\nUnbalance:\nGlioma: ', len_glioma)
    print('Meningioma: ', len_meningioma)
    print('Notumor: ', len_notumor)
    print('Pituitary: ', len_pituitary)
    print('Tot: ', len_glioma + len_meningioma + len_notumor + len_pituitary)
    
    rand_samples = random.sample(os_glioma, 2) + \
                   random.sample(os_meningioma, 2) + \
                   random.sample(os_notumor, 2) + \
                   random.sample(os_pituitary, 2)
    # Let's display mri with diagnoses
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
    norm_mean,norm_std = compute_img_mean_std(path + '/all images/')
    
    
    augment_transform=transforms.Compose([transforms.RandomRotation(10),                             # rotate +/- 10 degrees
                                          transforms.RandomHorizontalFlip(p=1.0),                    # reverse 50% of images
                                          transforms.RandomAffine(translate=(0.05,0.05), degrees=0),
                                          transforms.Resize(256),                                    # resize shortest side
                                          transforms.CenterCrop(256),                                # crop longest side
                                          # Inception_V3
                                          #transforms.Resize(299),                                    
                                          #transforms.CenterCrop(299),
                                          transforms.ToTensor(),
                                          transforms.Normalize(norm_mean, norm_std)
                                          ])
    
    ds_transform=transforms.Compose([transforms.Resize(256),                                         # resize shortest side
                                     transforms.CenterCrop(256),                                     # crop longest side
                                     # Inception_V3
                                     #transforms.Resize(299),                                         
                                     #transforms.CenterCrop(299),                                     
                                     transforms.ToTensor(),
                                     transforms.Normalize(norm_mean, norm_std)
                                     ])
    
    
    dataset_clean = ImageFolder(path +'/brain-tumor-mri-dataset', transform=ds_transform)
    dataset_augment = ImageFolder(path +'/brain-tumor-mri-dataset', transform=augment_transform)
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
    torch.save(train_dl, 'Brain_Mri_train_dl.pt')
    torch.save(val_dl, 'Brain_Mri_val_dl.pt')
    torch.save(test_dl, 'Brain_Mri_test_dl.pt')

# Function for plotting samples
def plot_samples(samples):  
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(15,8))
    for i in range(len(samples)):
        image = cv2.cvtColor(imread(samples[i]), cv2.COLOR_BGR2RGB)
        ax[i//4][i%4].imshow(image)
        if i==0 or i==4:
            ax[i//4][i%4].set_title("Glioma", fontsize=20)
        elif i==1 or i==5:
            ax[i//4][i%4].set_title("Meningioma", fontsize=20)
        elif i==2 or i==6:
            ax[i//4][i%4].set_title("NoTumor", fontsize=20)
        else:
            ax[i//4][i%4].set_title("Pituitary", fontsize=20)
        ax[i//4][i%4].axis('off')


# Compute mean and std for normalization
def compute_img_mean_std(image_paths):
    """
        computing the mean and std of three channel on the whole dataset,
        first we should normalize the image from 0-255 to 0-1
    """

    ext = ['png', 'jpg']    # Add image formats here
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
    return means, stdevs



def calculate_len_datasets(datasets):
    '''
    Parameters
    ----------
    datasets : 
        Array: [train_dataset, val_dataset, test_dataset]

    Returns
    -------
    all_len: 
        Array: [len_train_ds, len_glioma_train_ds, len_meningioma_train_ds, len_notumor_train_ds, len_pituitary_train_ds,
                len_val_ds, len_glioma_val_ds, len_meningioma_val_ds, len_notumor_val_ds, len_pituitary_val_ds
                len_test_ds, len_glioma_test_ds, len_meningioma_test_ds, len_notumor_test_ds, len_pituitary_test_ds]
    '''
    all_len = []
    for ds in datasets:
        glioma = 0
        meningioma = 0
        notumor = 0
        pituitary = 0
        for idx, (data, label) in enumerate(ds):
            if (label == 0):
                glioma = glioma + 1
            elif (label == 1):
                meningioma = meningioma + 1
            elif (label == 2):
                notumor = notumor + 1
            else:
                pituitary = pituitary + 1
        tot = len(ds)
        print('\nTot: ', tot)
        print('Glioma: ', glioma)
        print('Meningioma: ', meningioma)
        print('Notumor: ', notumor)
        print('Pituitary: ', pituitary)
        all_len.append(tot)
        all_len.append(glioma)
        all_len.append(meningioma)
        all_len.append(notumor)
        all_len.append(pituitary)
    return all_len


def get_loaders(datasets, all_len):
    '''
    ## Create a loader for each dataset in input() ##
    Parameters
    ----------
    datasets : 
        Array: [train_dataset, val_dataset, test_dataset]
    all_len : 
        Array: [len_train_ds, len_glioma_train_ds, len_meningioma_train_ds, len_notumor_train_ds, len_pituitary_train_ds,
                len_val_ds, len_glioma_val_ds, len_meningioma_val_ds, len_notumor_val_ds, len_pituitary_val_ds,
                len_test_ds, len_glioma_test_ds, len_meningioma_test_ds, len_notumor_test_ds, len_pituitary_test_ds]

    Returns
    -------
    loaders :
        Array: [train_loader, val_loader, test_loader]
    '''
    loaders = []
    x = 0
    for dataset in datasets:
        class_weights = [1.0/all_len[x+1], 1.0/all_len[x+2], 1.0/all_len[x+3], 1.0/all_len[x+4]]
        print('\nClass weights: ', class_weights)
        sample_weights = [0]*all_len[x]
        
        for idx, (data, label) in enumerate(dataset):
            class_weight = class_weights[label]
            sample_weights[idx] = class_weight

        sampler = WeightedRandomSampler(sample_weights, num_samples = all_len[x], replacement = True)
        loader = DataLoader(dataset, batch_size=128, sampler = sampler)
        loaders.append(loader)
        x = x + 5
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
    
