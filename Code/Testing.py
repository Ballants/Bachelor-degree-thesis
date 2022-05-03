import matplotlib.pyplot as plt

import time

import torch
import torch.nn.functional as F
import torchvision

from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report
 
from Pix2Pix_GAN import Generator

import torchattacks
import foolbox as fb

def Testing_model(tryDataset, tryModel, tryAttack, train=True):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #print(torch.cuda.is_available())
    print('Device: ', device)
    torch.cuda.empty_cache()
    
    ###############################################################################
    ## 5. TESTING ##
   
    # Loading model and Testing dataset
    if tryDataset == 'ChestX_ray':
        test_dl = torch.load('../ChestX_Ray_test_dl.pt')
        if tryModel == 'DenseNet121':
            model = torch.load('../ChestX_Ray DenseNet121.pth')
        elif tryModel == 'ResNet152':
            model = torch.load('../ChestX_Ray ResNet152.pth')
        elif tryModel == 'VGG19': 
            model = torch.load('../ChestX_Ray VGG19.pth')
        elif tryModel == 'AlexNet':
            model = torch.load('../ChestX_Ray AlexNet.pth')
        elif tryModel == 'MobileNet_V2':
            model = torch.load('../ChestX_Ray MobileNet_V2.pth')
        elif tryModel == 'Inception_V3':
            model = torch.load('../ChestX_Ray Inception_V3.pth')
    elif tryDataset == 'Brain_Mri':
        test_dl = torch.load('../Brain_Mri_test_dl.pt')
        if tryModel == 'DenseNet121':
            model = torch.load('../Brain_Mri DenseNet121.pth')
        elif tryModel == 'ResNet152':
            model = torch.load('../Brain_Mri ResNet152.pth')
        elif tryModel == 'VGG19': 
            model = torch.load('../Brain_Mri VGG19.pth')
        elif tryModel == 'AlexNet':
            model = torch.load('../Brain_Mri AlexNet.pth')
        elif tryModel == 'MobileNet_V2':
            model = torch.load('../Brain_Mri MobileNet_V2.pth')
        elif tryModel == 'Inception_V3':
            model = torch.load('../Brain_Mri Inception_V3.pth')
    
    # Loading GAN Generator 
    if tryModel == 'DenseNet121':
        if tryDataset == 'ChestX_ray':
            if tryAttack == 'FGSM':
                G = torch.load('Generator ChestX_Ray DenseNet121 FGSM.pth')
            elif tryAttack == 'BIM':
                G = torch.load('Generator ChestX_Ray DenseNet121 BIM.pth')
            elif tryAttack == 'DeepFool':
                G = torch.load('Generator ChestX_Ray DenseNet121 DeepFool.pth')
            elif tryAttack == 'PGD':
                G = torch.load('Generator ChestX_Ray DenseNet121 PGD.pth') 
        elif tryDataset == 'Brain_Mri':
            if tryAttack == 'FGSM':
                G = torch.load('Generator Brain_Mri DenseNet121 FGSM.pth')
            elif tryAttack == 'BIM':
                G = torch.load('Generator Brain_Mri DenseNet121 BIM.pth')
            elif tryAttack == 'DeepFool':
                G = torch.load('Generator Brain_Mri DenseNet121 DeepFool.pth')
            elif tryAttack == 'PGD':
                G = torch.load('Generator Brain_Mri DenseNet121 PGD.pth')
    elif tryModel == 'ResNet152':
        if tryDataset == 'ChestX_ray':
            if tryAttack == 'FGSM':
                G = torch.load('Generator ChestX_Ray ResNet152 FGSM.pth')
            elif tryAttack == 'BIM':
                G = torch.load('Generator ChestX_Ray ResNet152 BIM.pth')
            elif tryAttack == 'DeepFool':
                G = torch.load('Generator ChestX_Ray ResNet152 DeepFool.pth')
            elif tryAttack == 'PGD':
                G = torch.load('Generator ChestX_Ray ResNet152 PGD.pth') 
        elif tryDataset == 'Brain_Mri':
            if tryAttack == 'FGSM':
                G = torch.load('Generator Brain_Mri ResNet152 FGSM.pth')
            elif tryAttack == 'BIM':
                G = torch.load('Generator Brain_Mri ResNet152 BIM.pth')
            elif tryAttack == 'DeepFool':
                G = torch.load('Generator Brain_Mri ResNet152 DeepFool.pth')
            elif tryAttack == 'PGD':
                G = torch.load('Generator Brain_Mri ResNet152 PGD.pth')
    elif tryModel == 'VGG19':
        if tryDataset == 'ChestX_ray':
            if tryAttack == 'FGSM':
                G = torch.load('Generator ChestX_Ray VGG19 FGSM.pth')
            elif tryAttack == 'BIM':
                G = torch.load('Generator ChestX_Ray VGG19 BIM.pth')
            elif tryAttack == 'DeepFool':
                G = torch.load('Generator ChestX_Ray VGG19 DeepFool.pth')
            elif tryAttack == 'PGD':
                G = torch.load('Generator ChestX_Ray VGG19 PGD.pth') 
        elif tryDataset == 'Brain_Mri':
            if tryAttack == 'FGSM':
                G = torch.load('Generator Brain_Mri VGG19 FGSM.pth')
            elif tryAttack == 'BIM':
                G = torch.load('Generator Brain_Mri VGG19 BIM.pth')
            elif tryAttack == 'DeepFool':
                G = torch.load('Generator Brain_Mri VGG19 DeepFool.pth')
            elif tryAttack == 'PGD':
                G = torch.load('Generator Brain_Mri VGG19 PGD.pth')
    elif tryModel == 'MobileNet_V2':
        if tryDataset == 'ChestX_ray':
            if tryAttack == 'FGSM':
                G = torch.load('Generator ChestX_Ray MobileNet_V2 FGSM.pth')
            elif tryAttack == 'BIM':
                G = torch.load('Generator ChestX_Ray MobileNet_V2 BIM.pth')
            elif tryAttack == 'DeepFool':
                G = torch.load('Generator ChestX_Ray MobileNet_V2 DeepFool.pth')
            elif tryAttack == 'PGD':
                G = torch.load('Generator ChestX_Ray MobileNet_V2 PGD.pth') 
        elif tryDataset == 'Brain_Mri':
            if tryAttack == 'FGSM':
                G = torch.load('Generator Brain_Mri MobileNet_V2 FGSM.pth')
            elif tryAttack == 'BIM':
                G = torch.load('Generator Brain_Mri MobileNet_V2 BIM.pth')
            elif tryAttack == 'DeepFool':
                G = torch.load('Generator Brain_Mri MobileNet_V2 DeepFool.pth')
            elif tryAttack == 'PGD':
                G = torch.load('Generator Brain_Mri MobileNet_V2 PGD.pth')
    elif tryModel == 'Inception_V3':
        if tryDataset == 'ChestX_ray':
            if tryAttack == 'FGSM':
                G = torch.load('Generator ChestX_Ray Inception_V3 FGSM.pth')
            elif tryAttack == 'BIM':
                G = torch.load('Generator ChestX_Ray Inception_V3 BIM.pth')
            elif tryAttack == 'DeepFool':
                G = torch.load('Generator ChestX_Ray Inception_V3 DeepFool.pth')
            elif tryAttack == 'PGD':
                G = torch.load('Generator ChestX_Ray Inception_V3 PGD.pth') 
        elif tryDataset == 'Brain_Mri':
            if tryAttack == 'FGSM':
                G = torch.load('Generator Brain_Mri Inception_V3 FGSM.pth')
            elif tryAttack == 'BIM':
                G = torch.load('Generator Brain_Mri Inception_V3 BIM.pth')
            elif tryAttack == 'DeepFool':
                G = torch.load('Generator Brain_Mri Inception_V3 DeepFool.pth')
            elif tryAttack == 'PGD':
                G = torch.load('Generator Brain_Mri Inception_V3 PGD.pth')   
    
    G = G.eval()
    G.to(device)
    model = model.eval()
    
    # Setup attack
    if tryAttack == 'FGSM':
        atk = torchattacks.FGSM(model, eps=4/255)
    elif tryAttack == 'BIM':
        atk = torchattacks.BIM(model, eps=4/255, alpha=2/255, steps=100)
    elif tryAttack == 'DeepFool':
        atk = torchattacks.DeepFool(model)
    elif tryAttack == 'PGD':
        atk = torchattacks.PGD(model, eps=4/255, alpha=2/225, steps=100, random_start=True)
    
    ## Clean images ##
    preds_clean, labels_clean = test_predict(model, test_dl, device, atk, G)
    cm_clean = plotting_confusion_matrix(tryDataset, tryModel, tryAttack, labels_clean, preds_clean)
    performance_metrics(tryDataset, cm_clean, labels_clean, preds_clean)
    
    ## ATTACKING ##
    print('\nAttacco:', atk)
    preds_adv, labels_adv = test_predict(model, test_dl, device, atk, G, adv="adv")
    cm_adv = plotting_confusion_matrix(tryDataset, tryModel, tryAttack, labels_adv, preds_adv, adv="adv")
    performance_metrics(tryDataset, cm_adv, labels_adv, preds_adv)
    
    ## Mitigation with GAN ##
    preds_gan, labels_gan = test_predict(model, test_dl, device, atk, G, adv="gan")
    cm_gan = plotting_confusion_matrix(tryDataset, tryModel, tryAttack, labels_gan, preds_gan, adv="gan")
    performance_metrics(tryDataset, cm_gan, labels_gan, preds_gan)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1) 
    return torch.tensor(torch.sum(preds == labels).item() / len(preds)), preds

# The following three functions are needed to calculate metrics
def test_prediction(outputs):
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()           
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()             
    # combine predictions
    batch_preds = [pred for x in outputs for pred in x['preds'].tolist()] 
    # combine labels
    batch_labels = [lab for x in outputs for lab in x['labels'].tolist()]  
    
    return {'test_loss': epoch_loss.item(), 'test_acc': epoch_acc.item(),
            'test_preds': batch_preds, 'test_labels': batch_labels}  

def test_predict(model, test_dl, device, atk, G, adv="clean"):
    x=0
    outputs = []
    
    center_crop = torchvision.transforms.CenterCrop(224)
    
    since = time.time()
    for images, labels in test_dl:
        x = x+1
        
        labels = labels.to(device)
        images = images.to(device)
        
        if (adv == "adv"):
            images = center_crop(images)
            #print("adv: ", images.shape)
            adv_images = atk(images, labels)
            out = model(adv_images) 
            
            if x<10:
                #to plot images, perturbated images and noises
                fb.plot.images(images, n=1, scale=4.)   
                fb.plot.images(adv_images, n=1, scale=4.) 
                fb.plot.images(adv_images - images, n=1, bounds=(-0.1, 0.1), scale=4.)
                
            loss = F.cross_entropy(out, labels)                    
            acc,preds = accuracy(out, labels)
            
        elif (adv == "gan"):
            adv_images = atk(images, labels)
            
            gan_images = G(adv_images)
            
            images = center_crop(images)
            adv_images = center_crop(adv_images)
            gan_images = center_crop(gan_images)  
            #print("gan_images.shape: ", gan_images.shape)
            out = model(gan_images) 
            
            if x<3:
                #to plot images, perturbated images and noises
                fb.plot.images(images, n=1, scale=4.)   
                fb.plot.images(adv_images, n=1, scale=4.) 
                fb.plot.images(gan_images, n=1, scale=4.) 
            
            loss = F.cross_entropy(out, labels)                    
            acc,preds = accuracy(out, labels)
            
        elif (adv == "clean"):
            with torch.no_grad():
                '''
                if x<3:
                    #to plot images, perturbated images and noises
                    fb.plot.images(images, n=1, scale=4.)  
                '''
                images = center_crop(images)  
                out = model(images)
                '''
                if x<3:
                    #to plot images, perturbated images and noises
                    fb.plot.images(images, n=1, scale=4.)   
                '''   
                loss = F.cross_entropy(out, labels)                    
                acc,preds = accuracy(out, labels) 
            
        outputs.append({'val_loss': loss.detach(), 'val_acc':acc.detach(), 
                'preds':preds.detach(), 'labels':labels.detach()})
        
    results = test_prediction(outputs)                          
    print('\ntest_loss: {:.4f}, test_acc: {:.4f}'
          .format(results['test_loss'], results['test_acc']))
    
    print('\nTime: {}m {}s\n'.format((time.time()- since)//60, (time.time()- since)%60))
    
    return results['test_preds'], results['test_labels']

def plotting_confusion_matrix(tryDataset, tryModel, tryAttack, labels, preds, adv="clean"):
    # Plot confusion matrix
    cm  = confusion_matrix(labels, preds)
    plt.figure()
    plot_confusion_matrix(cm,figsize=(12,8), cmap=plt.cm.Blues, fontcolor_threshold=0.7)
    
    if tryDataset == 'ChestX_ray':
        plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
        plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
    elif tryDataset == 'Brain_Mri':
        plt.xticks(range(4), ['glioma', 'meningioma', 'notumor', 'pituitary'], fontsize=13)
        plt.yticks(range(4), ['glioma', 'meningioma', 'notumor', 'pituitary'], fontsize=13)
    
    plt.xlabel('Predicted Label', fontsize=18)
    plt.ylabel('True Label', fontsize=18)
    
    if (adv == "clean"):
        plt.suptitle('[{}] Confusion Matrix'
                     .format(tryModel),
                     fontsize=30)
    elif (adv == "adv"):
        plt.suptitle('[{} - {}] Confusion Matrix'
                     .format(tryModel, tryAttack),
                     fontsize=30)
    elif (adv == "gan"):
        plt.suptitle('[{} - {}] Confusion Matrix GAN'
                     .format(tryModel, tryAttack),
                     fontsize=30)
    plt.show()
    return cm

def performance_metrics(tryDataset, cm, labels, preds):
    # Next, let's calculate recall, precision and f1 score. This is one of the most key metrics for classification problems.
    # Compute Performance Metrics
    if tryDataset == 'ChestX_ray':
        classes = ['normal', 'pneumonia']
    elif tryDataset == 'Brain_Mri': 
        classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    report = classification_report(labels, preds, target_names=classes)
    print(report)


def main():
    
    ## ChestX_ray ##
    
    Testing_model('ChestX_ray', 'DenseNet121', 'FGSM')      
    Testing_model('ChestX_ray', 'DenseNet121', 'BIM')       
    Testing_model('ChestX_ray', 'DenseNet121', 'DeepFool')  
    Testing_model('ChestX_ray', 'DenseNet121', 'PGD')       
    
    Testing_model('ChestX_ray', 'ResNet152', 'FGSM')
    Testing_model('ChestX_ray', 'ResNet152', 'BIM')
    Testing_model('ChestX_ray', 'ResNet152', 'DeepFool')
    Testing_model('ChestX_ray', 'ResNet152', 'PGD')
    
    Testing_model('ChestX_ray', 'VGG19', 'FGSM')
    Testing_model('ChestX_ray', 'VGG19', 'BIM')
    Testing_model('ChestX_ray', 'VGG19', 'DeepFool')
    Testing_model('ChestX_ray', 'VGG19', 'PGD')
    
    Testing_model('ChestX_ray', 'MobileNet_V2', 'FGSM')
    Testing_model('ChestX_ray', 'MobileNet_V2', 'BIM')
    Testing_model('ChestX_ray', 'MobileNet_V2', 'DeepFool')
    Testing_model('ChestX_ray', 'MobileNet_V2', 'PGD')
    
    Testing_model('ChestX_ray', 'Inception_V3', 'FGSM')
    Testing_model('ChestX_ray', 'Inception_V3', 'BIM')
    Testing_model('ChestX_ray', 'Inception_V3', 'DeepFool')
    Testing_model('ChestX_ray', 'Inception_V3', 'PGD')
    
    ###########################################################################
    ## Brain_Mri ##
    
    Testing_model('Brain_Mri', 'DenseNet121', 'FGSM')
    Testing_model('Brain_Mri', 'DenseNet121', 'BIM')
    Testing_model('Brain_Mri', 'DenseNet121', 'DeepFool')
    Testing_model('Brain_Mri', 'DenseNet121', 'PGD')           
    
    Testing_model('Brain_Mri', 'ResNet152', 'FGSM')
    Testing_model('Brain_Mri', 'ResNet152', 'BIM')
    Testing_model('Brain_Mri', 'ResNet152', 'DeepFool')
    Testing_model('Brain_Mri', 'ResNet152', 'PGD')
    
    Testing_model('Brain_Mri', 'VGG19', 'FGSM')
    Testing_model('Brain_Mri', 'VGG19', 'BIM')
    Testing_model('Brain_Mri', 'VGG19', 'DeepFool')
    Testing_model('Brain_Mri', 'VGG19', 'PGD')
    
    Testing_model('Brain_Mri', 'MobileNet_V2', 'FGSM')
    Testing_model('Brain_Mri', 'MobileNet_V2', 'BIM')
    Testing_model('Brain_Mri', 'MobileNet_V2', 'DeepFool')
    Testing_model('Brain_Mri', 'MobileNet_V2', 'PGD')
    
    Testing_model('Brain_Mri', 'Inception_V3', 'FGSM')
    Testing_model('Brain_Mri', 'Inception_V3', 'BIM')
    Testing_model('Brain_Mri', 'Inception_V3', 'DeepFool')
    Testing_model('Brain_Mri', 'Inception_V3', 'PGD')
    
if __name__ == "__main__":
    main()
    
