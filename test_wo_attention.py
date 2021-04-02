from models_noa import Bokeh_Generator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
import time
from options import opt
from gauss import *
from torch.autograd import Variable

from canny import *

testing_epoch = opt.testing_epoch
testing_mode = opt.testing_mode

print("Checking for epoch:", testing_epoch)
print("Checking for mode : ", testing_mode)

CHECKPOINTS_DIR = opt.checkpoints_dir

if testing_mode == "Nat":
    print("Checking for Natural Images")
    HAZY_DIR = opt.testing_dir_nat
else:
    print("Checking for Synthetic Images")
    HAZY_DIR = opt.testing_dir_syn

result_dir = './EP'+str(testing_epoch)+'_'+HAZY_DIR.replace('.', '').replace('/', '_')+'_'+CHECKPOINTS_DIR.replace('.', '').replace('/', '_')+'_no_attention/'

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cpu'        

bokeh_net = Bokeh_Generator()
checkpoint = torch.load(os.path.join(CHECKPOINTS_DIR,'netG_'+ str(testing_epoch)+".pth"), map_location=torch.device('cpu'))
bokeh_net.load_state_dict(checkpoint['model_state_dict'])
bokeh_net.eval()
bokeh_net.to(device)

k = 2**(1/2)
sigma = 1.6
channels = 1
gaussian_kernel = 5

gauss_net_k_sigma = GaussianSmoothing(channels*3, gaussian_kernel, sigma)
gauss_net_k_sigma.to(device)

gauss_net_k3_sigma = GaussianSmoothing(channels*3, gaussian_kernel, (k**5)*sigma)
gauss_net_k3_sigma.to(device)

gauss_net_k5_sigma = GaussianSmoothing(channels*3, gaussian_kernel, (k**14)*sigma)
gauss_net_k5_sigma.to(device)

canny_net = CannyNet(threshold=20.0, use_cuda=False)
print('canny network', canny_net)
canny_net.to(device)



if __name__ =='__main__':

    total_files = os.listdir(HAZY_DIR)
    ready_samples = os.listdir(result_dir)
    st = time.time()

    for m in total_files:
    # m= "6.png"
        if str(m) in ready_samples:
            print('Already done {0}', str(m))
            continue
        print("Testing image ", str(m))

        #img = cv2.cvtColor(cv2.imread(HAZY_DIR + str(m)), cv2.COLOR_BGR2YCR_CB)
        img = cv2.imread(HAZY_DIR + str(m))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        img = img.astype(np.float32)
        h, w, c = img.shape
        img, cr, cb = cv2.split(img)
        
        img=img/255.0
        train_x = np.zeros((1, channels, h, w)).astype(np.float32)
        train_x[0,0,:,:] = img

        dataset_torchx = torch.from_numpy(train_x)
        dataset_torchx=dataset_torchx.to(device)

        gauss_canny_dataset_input = torch.cat((dataset_torchx, dataset_torchx, dataset_torchx), dim=1)

        guass_k_sigma = gauss_net_k_sigma(gauss_canny_dataset_input).to(device)
        guass_k3_sigma = gauss_net_k3_sigma(gauss_canny_dataset_input).to(device)
        guass_k5_sigma = gauss_net_k5_sigma(gauss_canny_dataset_input).to(device)

        canny_no_bokeh_batch = Variable(torch.empty([1, channels, h, w])).to(device)
        canny_no_bokeh_batch[0] = canny_net(gauss_canny_dataset_input[0].unsqueeze(0))

        output=bokeh_net(dataset_torchx, guass_k_sigma, guass_k3_sigma, guass_k5_sigma, canny_no_bokeh_batch)

        output=output*255.0
        output = output.cpu()
        a=output.detach().numpy()

        res = a[0,0,:,:]

        res = cv2.merge((np.uint8(res), np.uint8(cr), np.uint8(cb)))
        res = cv2.cvtColor(res, cv2.COLOR_YCR_CB2BGR)
        cv2.imwrite(result_dir + str(m),np.uint8(res))
        print('{')
        print('saved image ', str(m), ' at ', str(result_dir))
        print('image height ', str(res.shape[1]))
        print('image width ', str(res.shape[0]))
        print('}\n')

    end = time.time()
    print('Total time taken in secs : '+str(end-st))
    print('Per image (avg): '+ str(float((end-st)/len(total_files))))
