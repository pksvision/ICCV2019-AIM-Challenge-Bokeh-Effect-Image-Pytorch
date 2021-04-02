from train_canny import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
import time
import matplotlib.pyplot as plt

model_dirs = "./checkpoints/"
mse_list = []
start_epoch = 1
epochs_done = 50
loss_names = ['Mean-Squared Error']

for i in range(start_epoch, epochs_done+1):

    if os.path.exists(os.path.join(model_dirs,'netG_'+ str(i)+".pt")):
        filepath = os.path.join(model_dirs,'netG_'+ str(i)+".pt")
    else:
        filepath = os.path.join(model_dirs,'netG_'+ str(i)+".pth")

    # filepath = os.path.join(model_dirs,'netG_'+ str(i)+".pt") or os.path.join(model_dirs,'netG_'+ str(i)+".pth")
    checkpoint_g = 	torch.load(filepath)
    mse_list.append(float(checkpoint_g['total_mse_loss']))
	


x = [i+1 for i in range(start_epoch, epochs_done+1)]
y = [mse_list]
plt.xlabel("$Epochs$")
plt.ylabel("$Loss$")
plt.title("$Loss Graph$")
plt.grid()

for i in range(len(y)):
	plt.plot(x , [y[i][l] for l in range(len(y[i]))], label = "$"+str(loss_names[i])+"$")

plt.legend()
plt.savefig('NetG_MSE_Loss.eps')

for l in range(len(mse_list)):
	print(mse_list[l], l+1)

print(min(mse_list), mse_list.index(min(mse_list))+1)