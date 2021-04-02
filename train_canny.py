import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import dataset as dataset
import math
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from options import opt, device
from models import *
from misc import *
from canny import *
from gauss import *
from Sobel import *

if __name__ == '__main__':

    batches = int(opt.num_images / opt.batch_size)

    netG = Bokeh_Generator()
    print('bokeh network ', netG)
    mse_loss = nn.MSELoss()
    netG.to(device)
    optim_g = optim.Adam(netG.parameters(), 
                         lr=opt.learning_rate_g, 
                         betas = (opt.beta1, opt.beta2), 
                         weight_decay=opt.wd_g)
    

    # normalized input to canny and gaussians

    canny_net = CannyNet(threshold=20.0)
    print('canny network', canny_net)
    canny_net.to(device)
    for p in canny_net.parameters():
        p.requires_grad = False



    sobel_net = Sobel_Op()
    sobel_net.to(device)
    for p in sobel_net.parameters():
        p.requires_grad = False


    k = 2**(1/2)
    sigma = 1.6
    channels = 3
    gaussian_kernel = 5

    gauss_net_k_sigma = GaussianSmoothing(channels, gaussian_kernel, sigma)
    gauss_net_k_sigma.to(device)

    gauss_net_k3_sigma = GaussianSmoothing(channels, gaussian_kernel, (k**5)*sigma)
    gauss_net_k3_sigma.to(device)

    gauss_net_k5_sigma = GaussianSmoothing(channels, gaussian_kernel, (k**14)*sigma)
    gauss_net_k5_sigma.to(device)

    dataset = dataset.Dataset_Load(no_bokeh_path= opt.no_bokeh_dir, 
                                   bokeh_path= opt.bokeh_dir, 
                                   transform = dataset.ToTensor())

    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    
    if not os.path.exists(opt.checkpoints_dir):
        os.makedirs(opt.checkpoints_dir)    
    models_loaded = getLatestCheckpointName()
    latest_checkpoint_G = models_loaded
    print('loading model for generator ', latest_checkpoint_G)
    
    if latest_checkpoint_G == None:
    
        start_epoch = 1
        print('No checkpoints found for netG')
    
    else:
    
        checkpoint_g = torch.load(os.path.join(opt.checkpoints_dir, latest_checkpoint_G))
        start_epoch = checkpoint_g['epoch'] + 1
        netG.load_state_dict(checkpoint_g['model_state_dict'])
        optim_g.load_state_dict(checkpoint_g['optimizer_state_dict'])
        netG.train()
    
        print('Restoring model from checkpoint ' + str(start_epoch))
    
    for epoch in range(start_epoch, opt.end_epoch + 1):    

        opt.total_mse_loss = 0.0
        opt.total_sobel_loss = 0.0

        for i_batch, sample_batched in enumerate(dataloader):

            no_bokeh_batch = sample_batched['no_bokeh']
            bokeh_batch = sample_batched['bokeh']

            no_bokeh_batch = no_bokeh_batch.to(device)
            bokeh_batch = bokeh_batch.to(device)

            no_bokeh_batch_cats = torch.cat((no_bokeh_batch,no_bokeh_batch, no_bokeh_batch), dim=1)

            guass_k_sigma = gauss_net_k_sigma(no_bokeh_batch_cats).to(device)
            guass_k3_sigma = gauss_net_k3_sigma(no_bokeh_batch_cats).to(device)
            guass_k5_sigma = gauss_net_k5_sigma(no_bokeh_batch_cats).to(device)

            canny_no_bokeh_batch = Variable(torch.empty([opt.batch_size, opt.channels, opt.image_size, opt.image_size]), requires_grad=True).to(device)
            for batch in range(opt.batch_size):
                canny_no_bokeh_batch[batch] = canny_net(no_bokeh_batch_cats[batch].unsqueeze(0))

            for p in netG.parameters():
                p.requires_grad = True

            optim_g.zero_grad()

            # print(no_bokeh_batch.shape, "---------")
            # print(guass_k_sigma.shape, "---------")
            # print(guass_k3_sigma.shape, "---------")
            # print(guass_k5_sigma.shape, "---------")
            # print(canny_no_bokeh_batch.shape, "---------")
            
            pred_batch = netG(no_bokeh_batch, guass_k_sigma, guass_k3_sigma, guass_k5_sigma, canny_no_bokeh_batch)
            
            # print(pred_batch.shape, "------------")
            # print(bokeh_batch.shape, "------------")
            
            batch_mse_loss = mse_loss(pred_batch, bokeh_batch)
            batch_mse_loss.backward(retain_graph=True)   
            # print(netG.resnet.conv1[0].weight.grad[0][0])

            opt.batch_mse_loss = batch_mse_loss.item()  
            opt.total_mse_loss += opt.batch_mse_loss  

            sobel_pred = sobel_net(pred_batch)      
            sobel_target = sobel_net(bokeh_batch)

            sobel_loss = mse_loss(sobel_pred[0], sobel_target[0])
            sobel_loss += mse_loss(sobel_pred[1], sobel_target[1])

            sobel_loss = torch.mul(sobel_loss, opt.lambda_sobel)

            sobel_loss.backward()
            # print(netG.resnet.conv1[0].weight.grad[0][0])

            opt.batch_sobel_loss = sobel_loss.item()
            opt.total_sobel_loss += opt.batch_sobel_loss
            
            optim_g.step()

            print('training epoch %d, %d / %d patches are finished, g_mse = %.6f, g_sobel = %.6f' % (
             epoch, i_batch, batches, opt.batch_mse_loss, opt.batch_sobel_loss))

        torch.save({'epoch':epoch, 
                    'model_state_dict':netG.state_dict(), 
                    'optimizer_state_dict':optim_g.state_dict(), 
                    'opt':opt,
                    'total_mse_loss':opt.total_mse_loss,
                    'total_sobel_loss':opt.total_sobel_loss
                   }, os.path.join(opt.checkpoints_dir, 'netG_' + str(epoch) + '.pth'))