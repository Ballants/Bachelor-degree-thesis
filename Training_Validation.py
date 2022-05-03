import numpy as np 
import matplotlib.pyplot as plt
import copy
import time

import torch
import torch.nn as nn
import torchvision


def Training_and_Validation_model(tryDataset, tryModel):

    ## 1. DATA LOADING ##
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #print(torch.cuda.is_available())
    print('Device: ', device)
    torch.cuda.empty_cache()
    
    if tryDataset == 'ChestX_ray':
        train_dl = torch.load('../ChestX_ray_train_dl.pt')
        val_dl = torch.load('../ChestX_ray_val_dl.pt')
    elif tryDataset == 'Brain_Mri':
        train_dl = torch.load('../Brain_Mri_train_dl.pt')
        val_dl = torch.load('../Brain_Mri_val_dl.pt')
    
    loaders = {'train':train_dl, 'val':val_dl}
    dataset_sizes = {'train':len(train_dl.dataset), 'val':len(val_dl.dataset)}
    
    ###############################################################################
    ## 2. CREATING MODEL ##
    
    print('\nTryModel: ', tryModel)
    
    if tryDataset == 'ChestX_ray':
        num_classes = 2
    elif tryDataset == 'Brain_Mri':
        num_classes = 4
        
    if tryModel == 'DenseNet121':
        model = torchvision.models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif tryModel == 'ResNet152':
        model = torchvision.models.resnet152(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif tryModel == 'VGG19':
        model = torchvision.models.vgg19(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif tryModel == 'MobileNet_V2':
        model = torchvision.models.mobilenet_v2(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif tryModel == 'Inception_V3':
        model = torchvision.models.inception_v3(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.aux_logits=False
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    ###############################################################################
    ## 3. TRAINING  ##
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 4, gamma=0.1)
    
    model.to(device)
    epochs = 50
    model, losses, accuracies = train(model, criterion, optimizer, scheduler, epochs, loaders, device, dataset_sizes)
    
    # Saving model
    if tryModel == 'DenseNet121':
        if tryDataset == 'ChestX_ray':
            torch.save(model, 'ChestX_Ray DenseNet121.pth')
        elif tryDataset == 'Brain_Mri':
            torch.save(model, 'Brain_Mri DenseNet121.pth')
    elif tryModel == 'ResNet152':
        if tryDataset == 'ChestX_ray':
            torch.save(model, 'ChestX_Ray ResNet152.pth')
        elif tryDataset == 'Brain_Mri':
            torch.save(model, 'Brain_Mri ResNet152.pth')
    elif tryModel == 'VGG19':
        if tryDataset == 'ChestX_ray':
            torch.save(model, 'ChestX_Ray VGG19.pth')
        elif tryDataset == 'Brain_Mri':
            torch.save(model, 'Brain_Mri VGG19.pth')
    elif tryModel == 'MobileNet_V2':
        if tryDataset == 'ChestX_ray':
            torch.save(model, 'ChestX_Ray MobileNet_V2.pth')
        elif tryDataset == 'Brain_Mri':
            torch.save(model, 'Brain_Mri MobileNet_V2.pth')
    elif tryModel == 'Inception_V3':
        if tryDataset == 'ChestX_ray':
            torch.save(model, 'ChestX_Ray Inception_V3.pth')
        elif tryDataset == 'Brain_Mri':
            torch.save(model, 'Brain_Mri Inception_V3.pth')
        
    ###############################################################################
    ## 4. PLOT ACCURACY AND LOSS ##
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    t = f.suptitle('[{}] Performance'.format(tryModel), fontsize=18)
    f.subplots_adjust(top=0.85, wspace=0.3)
    
    for i in range(len(accuracies['train'])):
        accuracies['train'][i] = accuracies['train'][i].cpu().detach().numpy()  
    for i in range(len(accuracies['val'])):
        accuracies['val'][i] = accuracies['val'][i].cpu().detach().numpy()
    
    epoch_list = range(epochs)
    ax1.plot(epoch_list, accuracies['train'], label='Train Accuracy')
    ax1.plot(epoch_list, accuracies['val'], label='Validation Accuracy')
    ax1.set_xticks(np.arange(0, epochs+1, 5))
    ax1.set_ylabel('Accuracy Value')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Accuracy')
    l1 = ax1.legend(loc="best")
    
    ax2.plot(epoch_list, losses['train'], label='Train Loss')
    ax2.plot(epoch_list, losses['val'], label='Validation Loss')
    ax2.set_xticks(np.arange(0, epochs+1, 5))
    ax2.set_ylabel('Loss Value')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Loss')
    l2 = ax2.legend(loc="best")

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1) 
    return torch.tensor(torch.sum(preds == labels).item() / len(preds)), preds

# In the optimizer we specify only the classifier. Set optimizer, only train the classifier parameters, feature parameters are frozen
def train(model, criterion, optimizer, scheduler, epochs, loaders, device, dataset_sizes):
  
    #save the losses for further visualization
    losses = {'train':[], 'val':[]}
    accuracies = {'train':[], 'val':[]}
  
    since = time.time()
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(epochs):
      
      for phase in ['train', 'val']:
          
        if phase == 'train':
          model.train()
        else:
          model.eval()
        
        running_loss = 0.0
        running_corrects = 0.0
        
        for inputs, labels in loaders[phase]:
          inputs, labels = inputs.to(device), labels.to(device)
          
          if (model != 'Inception_V3'):
              center_crop = torchvision.transforms.CenterCrop(224)
              inputs = center_crop(inputs)
             
          optimizer.zero_grad()
  
          with torch.set_grad_enabled(phase=='train'):
            outp = model(inputs)
            _, pred = torch.max(outp, 1)
            loss = criterion(outp, labels)
          
            if phase == 'train':
              loss.backward()
              optimizer.step()
          
          running_loss += loss.item()*inputs.size(0)
          running_corrects += torch.sum(pred == labels.data)
  
  
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double()/dataset_sizes[phase]
        losses[phase].append(epoch_loss)
        accuracies[phase].append(epoch_acc)
        
        if phase == 'train':
          print('Epoch: {}/{}'.format(epoch+1, epochs))
        print('{}:\n- loss: {}\n- accuracy: {}'.format(phase, epoch_loss, epoch_acc))
      
        if phase == 'val':
          print('Time: {}m {}s'.format((time.time()- since)//60, (time.time()- since)%60))
        
        if phase == 'val' and epoch_acc > best_acc:
          best_acc = epoch_acc
          best_model = copy.deepcopy(model.state_dict())
      scheduler.step()  
    time_elapsed = time.time() - since
    print('Training Time: {}m {}s'.format(time_elapsed//60, time_elapsed%60)) 
    print('Best accuracy: {}'.format(best_acc))
  
    model.load_state_dict(best_model)
    return model, losses, accuracies

def main():
    
    ## ChestX_ray ##
    
    Training_and_Validation_model('ChestX_ray', 'DenseNet121')
    Training_and_Validation_model('ChestX_ray', 'ResNet152')
    Training_and_Validation_model('ChestX_ray', 'VGG19')
    Training_and_Validation_model('ChestX_ray', 'MobileNet_V2')
    Training_and_Validation_model('ChestX_ray', 'Inception_V3')
    
    ###########################################################################
    ## Brain_Mri ##
    
    Training_and_Validation_model('Brain_Mri', 'DenseNet121')           
    Training_and_Validation_model('Brain_Mri', 'ResNet152')
    Training_and_Validation_model('Brain_Mri', 'VGG19')
    Training_and_Validation_model('Brain_Mri', 'MobileNet_V2')
    Training_and_Validation_model('Brain_Mri', 'Inception_V3')
    
if __name__ == "__main__":
    main()