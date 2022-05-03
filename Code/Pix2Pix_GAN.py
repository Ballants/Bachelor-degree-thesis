import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

import time

import foolbox as fb 
import torchattacks


def cnn_block(in_channels, out_channels, kernel_size, stride=1, padding=0, first_layer = False):
  
   if first_layer:
       return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
   else:
       return nn.Sequential(
           nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding), 
           nn.BatchNorm2d(out_channels, momentum=0.1, eps=1e-5), 
           )


def tcnn_block(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, first_layer = False):
   if first_layer:
       return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)

   else:
       return nn.Sequential(
           nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding), 
           nn.BatchNorm2d(out_channels, momentum=0.1, eps=1e-5), 
           )
   
    

class Generator(nn.Module):
    def __init__(self, c_dim = 3, gf_dim = 64, instance_norm=False):          # input: 256 x 256
        super(Generator, self).__init__()
        self.e1 = cnn_block(c_dim, gf_dim, 4, 2, 1, first_layer = True)       # 128 x 128
        self.e2 = cnn_block(gf_dim, gf_dim*2, 4, 2, 1,)                       # 64 x 64
        self.e3 = cnn_block(gf_dim*2, gf_dim*4, 4, 2, 1,)                     # 32 x 32
        self.e4 = cnn_block(gf_dim*4, gf_dim*8, 4, 2, 1,)                     # 16 x 16
        self.e5 = cnn_block(gf_dim*8, gf_dim*8, 4, 2, 1,)                     # 8 x 8
        self.e6 = cnn_block(gf_dim*8, gf_dim*8, 4, 2, 1,)                     # 4 x 4
        self.e7 = cnn_block(gf_dim*8, gf_dim*8, 4, 2, 1,)                     # 2 x 2
        self.e8 = cnn_block(gf_dim*8, gf_dim*8, 4, 2, 1, first_layer=True)    # 1 x 1
        
        self.d1 = tcnn_block(gf_dim*8, gf_dim*8, 4, 2, 1)                     # 2 x 2
        self.d2 = tcnn_block(gf_dim*8*2, gf_dim*8, 4, 2, 1)                   # 4 x 4
        self.d3 = tcnn_block(gf_dim*8*2, gf_dim*8, 4, 2, 1)                   # 8 x 8
        self.d4 = tcnn_block(gf_dim*8*2, gf_dim*8, 4, 2, 1)                   # 16 x 16
        self.d5 = tcnn_block(gf_dim*8*2, gf_dim*4, 4, 2, 1)                   # 32 x 32
        self.d6 = tcnn_block(gf_dim*4*2, gf_dim*2, 4, 2, 1)                   # 64 x 64
        self.d7 = tcnn_block(gf_dim*2*2, gf_dim*1, 4, 2, 1)                   # 128 x 128
        self.d8 = tcnn_block(gf_dim*1*2, c_dim, 4, 2, 1, first_layer = True)  # 256 x 256
        self.tanh = nn.Tanh()

    def forward(self, x):
        #print("\nGenerator:\nx: ", x.shape)
        e1 = self.e1(x)
        #print("e1: ", e1.shape)
        e2 = self.e2(F.leaky_relu(e1, 0.2))
        #print("2: ", e2.shape)
        e3 = self.e3(F.leaky_relu(e2, 0.2))
        #print("3: ", e3.shape)
        e4 = self.e4(F.leaky_relu(e3, 0.2))
        #print("4: ", e4.shape)
        e5 = self.e5(F.leaky_relu(e4, 0.2))
        #print("5: ", e5.shape)
        e6 = self.e6(F.leaky_relu(e5, 0.2))
        #print("6: ", e6.shape)
        e7 = self.e7(F.leaky_relu(e6, 0.2))
        #print("7: ", e7.shape)
        e8 = self.e8(F.leaky_relu(e7, 0.2))
        #print("8: ", e8.shape)
        d1 = torch.cat([F.dropout(self.d1(F.relu(e8)), 0.5, training=True), e7], 1)
        #print("d1: ", d1.shape)
        d2 = torch.cat([F.dropout(self.d2(F.relu(d1)), 0.5, training=True), e6], 1)
        #print("d2: ", d2.shape)
        d3 = torch.cat([F.dropout(self.d3(F.relu(d2)), 0.5, training=True), e5], 1)
        #print("3: ", d3.shape)
        d4 = torch.cat([self.d4(F.relu(d3)), e4], 1)
        #print("4: ", d4.shape)
        d5 = torch.cat([self.d5(F.relu(d4)), e3], 1)
        #print("5: ", d5.shape)
        d6 = torch.cat([self.d6(F.relu(d5)), e2], 1)
        #print("6: ", d6.shape)
        d7 = torch.cat([self.d7(F.relu(d6)), e1], 1)
        #print("7: ", d7.shape)
        d8 = self.d8(F.relu(d7))
        #print("8: ", d8.shape)
        return self.tanh(d8)
    
class Discriminator(nn.Module):
    def __init__(self, c_dim = 3, df_dim = 64, instance_norm=False):          # input: 256x256
        super(Discriminator, self).__init__()
        self.conv1 = cnn_block(c_dim*2, df_dim, 4, 2, 1, first_layer=True)    # 128 x 128
        self.conv2 = cnn_block(df_dim, df_dim*2, 4, 2, 1)                     # 64 x 64
        
        self.conv3 = cnn_block(df_dim*2, df_dim*4, 4, 2, 1)                   # 32 x 32
        self.conv4 = cnn_block(df_dim*4, df_dim*8, 4, 1, 1)                   # 31 x 31
        self.conv5 = cnn_block(df_dim*8, 1, 4, 1, 1, first_layer=True)        # 30 x 30
        
        #self.c = cnn_block(df_dim*2, 1, 2, 2, 2, first_layer=True)
        self.sigmoid = nn.Sigmoid()
    
    
    def forward(self, x, y):
        #print("\nDiscriminator\n: ", x.shape)
        O = torch.cat([x, y], dim=1)
        #print("1: ", O.shape)
        O = F.leaky_relu(self.conv1(O), 0.2)
        #print("2: ", O.shape)
        O = F.leaky_relu(self.conv2(O), 0.2)
        #print("3: ", O.shape)
        
        O = F.leaky_relu(self.conv3(O), 0.2)
        #print("4: ", O.shape)
        O = F.leaky_relu(self.conv4(O), 0.2)
        #print("5: ", O.shape)
        O = self.conv5(O)
        #print("6: ", O.shape)
        
        #O = F.leaky_relu(self.c(O), 0.2)
        #print("4: ", O.shape)
        return self.sigmoid(O)

def Training_GAN(tryDataset, tryModel, tryAttack):

    if tryDataset == 'ChestX_ray':
        train_loader = torch.load('../ChestX_Ray_train_dl.pt')
        if tryModel == 'DenseNet121':
            model = torch.load('../ChestX_Ray DenseNet121.pth')
        elif tryModel == 'ResNet152':
            model = torch.load('../ChestX_Ray ResNet152.pth')
        elif tryModel == 'VGG19': 
            model = torch.load('../ChestX_Ray VGG19.pth')
        elif tryModel == 'MobileNet_V2':
            model = torch.load('../ChestX_Ray MobileNet_V2.pth')
        elif tryModel == 'Inception_V3':
            model = torch.load('../ChestX_Ray Inception_V3.pth')
    elif tryDataset == 'Brain_Mri':
        train_loader = torch.load('../Brain_Mri_train_dl.pt')
        if tryModel == 'DenseNet121':
            model = torch.load('../Brain_Mri DenseNet121.pth')
        elif tryModel == 'ResNet152':
            model = torch.load('../Brain_Mri ResNet152.pth')
        elif tryModel == 'VGG19': 
            model = torch.load('../Brain_Mri VGG19.pth')
        elif tryModel == 'MobileNet_V2':
            model = torch.load('../Brain_Mri MobileNet_V2.pth')
        elif tryModel == 'Inception_V3':
            model = torch.load('../Brain_Mri Inception_V3.pth')
    
    # Setup attack
    if tryAttack == 'FGSM':
        atk = torchattacks.FGSM(model, eps=4/255)
    elif tryAttack == 'BIM':
        atk = torchattacks.BIM(model, eps=4/255, alpha=2/255, steps=100)
    elif tryAttack == 'DeepFool':
        atk = torchattacks.DeepFool(model)
    elif tryAttack == 'PGD':
        atk = torchattacks.PGD(model, eps=4/255, alpha=2/225, steps=100, random_start=True)
    
    
    # Define parameters
    epochs = 50
    
    gf_dim = 64
    df_dim = 64
    c_dim = 3
    
    L1_lambda = 100.0
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(torch.cuda.is_available())
    print('Device: ', device)
    torch.cuda.empty_cache()
    
    G = Generator(c_dim = c_dim, gf_dim = gf_dim).to(device)
    D = Discriminator(c_dim = c_dim, df_dim = df_dim).to(device)
    
    G_optimizer = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
    bce_criterion = nn.BCELoss()
    L1_criterion = nn.L1Loss()
    
    for ep in range(epochs):
        since = time.time()
        for i, data in enumerate(train_loader):  
            
            images, labels = data
            labels = labels.to(device)
            y = images.to(device)
            x = atk(y, labels).to(device)
            
            b_size = x.shape[0]
             
            real_class = torch.ones(b_size, 1, 30, 30).to(device)
            fake_class = torch.zeros(b_size, 1, 30, 30).to(device)
             
            #Train D
            D.zero_grad()
           
            real_patch = D(y, x)
            real_gan_loss=bce_criterion(real_patch, real_class)
             
            fake=G(x)
           
            fake_patch = D(fake.detach(), x)
            fake_gan_loss=bce_criterion(fake_patch, fake_class)
             
            D_loss = real_gan_loss + fake_gan_loss
            D_loss.backward()
            D_optimizer.step()
             
            if (i == 0):
                fb.plot.images(y, n=3, scale=4.)
                fb.plot.images(x, n=3, scale=4.)
                fb.plot.images(fake, n=3, scale=4.)   
             
            #Train G
            G.zero_grad()
            fake_patch = D(fake, x)
            fake_gan_loss=bce_criterion(fake_patch, real_class)
             
            L1_loss = L1_criterion(fake, y)
            G_loss = fake_gan_loss + L1_lambda * L1_loss
            G_loss.backward()
           
            G_optimizer.step()
            if (i == 0):
              print('Epoch [{}/{}],\tStep [{}/{}],\td_loss: {:.4f},\tg_loss: {:.4f},\tD(real): {:.2f},\tD(fake):{:.2f},\tg_loss_gan:{:.4f},\tg_loss_L1:{:.4f}'
                    .format(ep, epochs, i+1, len(train_loader), D_loss.item(), G_loss.item(), real_patch.mean(), fake_patch.mean(), fake_gan_loss.item(), L1_loss.item()))
            
        print('Time: {}m {}s'.format((time.time()- since)//60, (time.time()- since)%60))
    
    # Saving Generator
    if tryModel == 'DenseNet121':
        if tryDataset == 'ChestX_ray':
            if tryAttack == 'FGSM':
                torch.save(G, 'Generator ChestX_Ray DenseNet121 FGSM.pth')
            elif tryAttack == 'BIM':
                torch.save(G, 'Generator ChestX_Ray DenseNet121 BIM.pth')
            elif tryAttack == 'DeepFool':
                torch.save(G, 'Generator ChestX_Ray DenseNet121 DeepFool.pth')
            elif tryAttack == 'PGD':
                torch.save(G, 'Generator ChestX_Ray DenseNet121 PGD.pth') 
        elif tryDataset == 'Brain_Mri':
            if tryAttack == 'FGSM':
                torch.save(G, 'Generator Brain_Mri DenseNet121 FGSM.pth')
            elif tryAttack == 'BIM':
                torch.save(G, 'Generator Brain_Mri DenseNet121 BIM.pth')
            elif tryAttack == 'DeepFool':
                torch.save(G, 'Generator Brain_Mri DenseNet121 DeepFool.pth')
            elif tryAttack == 'PGD':
                torch.save(G, 'Generator Brain_Mri DenseNet121 PGD.pth')
    elif tryModel == 'ResNet152':
        if tryDataset == 'ChestX_ray':
            if tryAttack == 'FGSM':
                torch.save(G, 'Generator ChestX_Ray ResNet152 FGSM.pth')
            elif tryAttack == 'BIM':
                torch.save(G, 'Generator ChestX_Ray ResNet152 BIM.pth')
            elif tryAttack == 'DeepFool':
                torch.save(G, 'Generator ChestX_Ray ResNet152 DeepFool.pth')
            elif tryAttack == 'PGD':
                torch.save(G, 'Generator ChestX_Ray ResNet152 PGD.pth') 
        elif tryDataset == 'Brain_Mri':
            if tryAttack == 'FGSM':
                torch.save(G, 'Generator Brain_Mri ResNet152 FGSM.pth')
            elif tryAttack == 'BIM':
                torch.save(G, 'Generator Brain_Mri ResNet152 BIM.pth')
            elif tryAttack == 'DeepFool':
                torch.save(G, 'Generator Brain_Mri ResNet152 DeepFool.pth')
            elif tryAttack == 'PGD':
                torch.save(G, 'Generator Brain_Mri ResNet152 PGD.pth')
    elif tryModel == 'VGG19':
        if tryDataset == 'ChestX_ray':
            if tryAttack == 'FGSM':
                torch.save(G, 'Generator ChestX_Ray VGG19 FGSM.pth')
            elif tryAttack == 'BIM':
                torch.save(G, 'Generator ChestX_Ray VGG19 BIM.pth')
            elif tryAttack == 'DeepFool':
                torch.save(G, 'Generator ChestX_Ray VGG19 DeepFool.pth')
            elif tryAttack == 'PGD':
                torch.save(G, 'Generator ChestX_Ray VGG19 PGD.pth') 
        elif tryDataset == 'Brain_Mri':
            if tryAttack == 'FGSM':
                torch.save(G, 'Generator Brain_Mri VGG19 FGSM.pth')
            elif tryAttack == 'BIM':
                torch.save(G, 'Generator Brain_Mri VGG19 BIM.pth')
            elif tryAttack == 'DeepFool':
                torch.save(G, 'Generator Brain_Mri VGG19 DeepFool.pth')
            elif tryAttack == 'PGD':
                torch.save(G, 'Generator Brain_Mri VGG19 PGD.pth')
    elif tryModel == 'MobileNet_V2':
        if tryDataset == 'ChestX_ray':
            if tryAttack == 'FGSM':
                torch.save(G, 'Generator ChestX_Ray MobileNet_V2 FGSM.pth')
            elif tryAttack == 'BIM':
                torch.save(G, 'Generator ChestX_Ray MobileNet_V2 BIM.pth')
            elif tryAttack == 'DeepFool':
                torch.save(G, 'Generator ChestX_Ray MobileNet_V2 DeepFool.pth')
            elif tryAttack == 'PGD':
                torch.save(G, 'Generator ChestX_Ray MobileNet_V2 PGD.pth') 
        elif tryDataset == 'Brain_Mri':
            if tryAttack == 'FGSM':
                torch.save(G, 'Generator Brain_Mri MobileNet_V2 FGSM.pth')
            elif tryAttack == 'BIM':
                torch.save(G, 'Generator Brain_Mri MobileNet_V2 BIM.pth')
            elif tryAttack == 'DeepFool':
                torch.save(G, 'Generator Brain_Mri MobileNet_V2 DeepFool.pth')
            elif tryAttack == 'PGD':
                torch.save(G, 'Generator Brain_Mri MobileNet_V2 PGD.pth')
    elif tryModel == 'Inception_V3':
        if tryDataset == 'ChestX_ray':
            if tryAttack == 'FGSM':
                torch.save(G, 'Generator ChestX_Ray Inception_V3 FGSM.pth')
            elif tryAttack == 'BIM':
                torch.save(G, 'Generator ChestX_Ray Inception_V3 BIM.pth')
            elif tryAttack == 'DeepFool':
                torch.save(G, 'Generator ChestX_Ray Inception_V3 DeepFool.pth')
            elif tryAttack == 'PGD':
                torch.save(G, 'Generator ChestX_Ray Inception_V3 PGD.pth') 
        elif tryDataset == 'Brain_Mri':
            if tryAttack == 'FGSM':
                torch.save(G, 'Generator Brain_Mri Inception_V3 FGSM.pth')
            elif tryAttack == 'BIM':
                torch.save(G, 'Generator Brain_Mri Inception_V3 BIM.pth')
            elif tryAttack == 'DeepFool':
                torch.save(G, 'Generator Brain_Mri Inception_V3 DeepFool.pth')
            elif tryAttack == 'PGD':
                torch.save(G, 'Generator Brain_Mri Inception_V3 PGD.pth') 

def main():
    
    ## ChestX_ray ##
    
    Training_GAN('ChestX_ray', 'DenseNet121', 'FGSM')      
    Training_GAN('ChestX_ray', 'DenseNet121', 'BIM')       
    Training_GAN('ChestX_ray', 'DenseNet121', 'DeepFool')  
    Training_GAN('ChestX_ray', 'DenseNet121', 'PGD')       
    
    Training_GAN('ChestX_ray', 'ResNet152', 'FGSM')
    Training_GAN('ChestX_ray', 'ResNet152', 'BIM')
    Training_GAN('ChestX_ray', 'ResNet152', 'DeepFool')
    Training_GAN('ChestX_ray', 'ResNet152', 'PGD')
    
    Training_GAN('ChestX_ray', 'VGG19', 'FGSM')
    Training_GAN('ChestX_ray', 'VGG19', 'BIM')
    Training_GAN('ChestX_ray', 'VGG19', 'DeepFool')
    Training_GAN('ChestX_ray', 'VGG19', 'PGD')
    
    Training_GAN('ChestX_ray', 'MobileNet_V2', 'FGSM')
    Training_GAN('ChestX_ray', 'MobileNet_V2', 'BIM')
    Training_GAN('ChestX_ray', 'MobileNet_V2', 'DeepFool')
    Training_GAN('ChestX_ray', 'MobileNet_V2', 'PGD')
    
    Training_GAN('ChestX_ray', 'Inception_V3', 'FGSM')
    Training_GAN('ChestX_ray', 'Inception_V3', 'BIM')
    Training_GAN('ChestX_ray', 'Inception_V3', 'DeepFool')
    Training_GAN('ChestX_ray', 'Inception_V3', 'PGD')
    
    ###########################################################################
    ## Brain_Mri ##
    
    Training_GAN('Brain_Mri', 'DenseNet121', 'FGSM')
    Training_GAN('Brain_Mri', 'DenseNet121', 'BIM')
    Training_GAN('Brain_Mri', 'DenseNet121', 'DeepFool')
    Training_GAN('Brain_Mri', 'DenseNet121', 'PGD')           
    
    Training_GAN('Brain_Mri', 'ResNet152', 'FGSM')
    Training_GAN('Brain_Mri', 'ResNet152', 'BIM')
    Training_GAN('Brain_Mri', 'ResNet152', 'DeepFool')
    Training_GAN('Brain_Mri', 'ResNet152', 'PGD')
    
    Training_GAN('Brain_Mri', 'VGG19', 'FGSM')
    Training_GAN('Brain_Mri', 'VGG19', 'BIM')
    Training_GAN('Brain_Mri', 'VGG19', 'DeepFool')
    Training_GAN('Brain_Mri', 'VGG19', 'PGD')
    
    Training_GAN('Brain_Mri', 'MobileNet_V2', 'FGSM')
    Training_GAN('Brain_Mri', 'MobileNet_V2', 'BIM')
    Training_GAN('Brain_Mri', 'MobileNet_V2', 'DeepFool')
    Training_GAN('Brain_Mri', 'MobileNet_V2', 'PGD')
    
    Training_GAN('Brain_Mri', 'Inception_V3', 'FGSM')
    Training_GAN('Brain_Mri', 'Inception_V3', 'BIM')
    Training_GAN('Brain_Mri', 'Inception_V3', 'DeepFool')
    Training_GAN('Brain_Mri', 'Inception_V3', 'PGD')
    
if __name__ == "__main__":
    main()
