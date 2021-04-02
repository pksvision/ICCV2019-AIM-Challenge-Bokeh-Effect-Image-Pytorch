import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import cv2
import os
import numpy as np
import argparse
from options import opt
from Self_Attention import Self_Attn
from resnet import *

class Bokeh_Generator(nn.Module):

    def __init__(self):
        
        super(Bokeh_Generator, self).__init__()

        self.conv1_l1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1_l1 = nn.BatchNorm2d(num_features=64)
        self.relu1_l1 = nn.ReLU()

        self.conv2_l1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2_l1 = nn.BatchNorm2d(num_features=64)
        self.relu2_l1 = nn.ReLU()

        self.conv3_l1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3_l1 = nn.BatchNorm2d(num_features=64)
        self.relu3_l1 = nn.ReLU()

        self.conv4_l1 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.bn4_l1 = nn.BatchNorm2d(num_features=3)
        self.relu4_l1 = nn.ReLU()


        self.conv1_l2 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1_l2 = nn.BatchNorm2d(num_features=64)
        self.relu1_l2 = nn.ReLU()

        self.conv2_l2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2_l2 = nn.BatchNorm2d(num_features=64)
        self.relu2_l2 = nn.ReLU()

        self.conv3_l2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3_l2 = nn.BatchNorm2d(num_features=64)
        self.relu3_l2 = nn.ReLU()

        self.conv4_l2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.bn4_l2 = nn.BatchNorm2d(num_features=3)
        self.relu4_l2 = nn.ReLU()


        self.conv1_l3 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1_l3 = nn.BatchNorm2d(num_features=64)
        self.relu1_l3 = nn.ReLU()

        self.conv2_l3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2_l3 = nn.BatchNorm2d(num_features=64)
        self.relu2_l3 = nn.ReLU()

        self.conv3_l3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3_l3 = nn.BatchNorm2d(num_features=64)
        self.relu3_l3 = nn.ReLU()

        self.conv4_l3 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.bn4_l3 = nn.BatchNorm2d(num_features=3)
        self.relu4_l3 = nn.ReLU()

        self.attn = Self_Attn(9, 'relu')

        self.resnet = ResNet18()


    def forward(self, input, input_k_sigma, input_k3_sigma, input_k5_sigma, input_canny):


        input_k_sigma = F.pad(input_k_sigma, (2, 2, 2, 2), mode='reflect')
        input_k3_sigma = F.pad(input_k3_sigma, (2, 2, 2, 2), mode='reflect')
        input_k5_sigma = F.pad(input_k5_sigma, (2, 2, 2, 2), mode='reflect')

        input_layer_1 = torch.cat((input_k_sigma, input_canny), dim=1)
        input_layer_2 = torch.cat((input_k3_sigma, input_canny), dim=1)
        input_layer_3 = torch.cat((input_k5_sigma, input_canny), dim=1)

        inp_1_l1 = self.relu1_l1(self.bn1_l1(self.conv1_l1(input_layer_1)))
        inp_2_l1 = self.relu2_l1(self.bn2_l1(self.conv2_l1(inp_1_l1)))
        inp_3_l1 = self.relu3_l1(self.bn3_l1(self.conv3_l1(inp_2_l1)))
        op_l1 = self.relu4_l1(self.bn4_l1(self.conv4_l1(inp_3_l1)))

        inp_1_l2 = self.relu1_l2(self.bn1_l2(self.conv1_l2(input_layer_2)))
        inp_2_l2 = self.relu2_l2(self.bn2_l2(self.conv2_l2(inp_1_l2)))
        inp_3_l2 = self.relu3_l2(self.bn3_l2(self.conv3_l2(inp_2_l2)))
        op_l2 = self.relu4_l2(self.bn4_l2(self.conv4_l2(inp_3_l2)))

        inp_1_l3 = self.relu1_l3(self.bn1_l3(self.conv1_l3(input_layer_3)))
        inp_2_l3 = self.relu2_l3(self.bn2_l3(self.conv2_l3(inp_1_l3)))
        inp_3_l3 = self.relu3_l3(self.bn3_l3(self.conv3_l3(inp_2_l3)))
        op_l3 = self.relu4_l3(self.bn4_l3(self.conv4_l3(inp_3_l3)))

        input_resnet = torch.cat((op_l1, op_l2, op_l3), dim=1)
        input_resnet = self.attn(input_resnet)

        output = self.resnet(input_resnet)

        output = output + input

        return output



class Discriminator(nn.Module):

    def __init__(self):

        super(Discriminator, self).__init__()
        
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.bn1_1 = nn.BatchNorm2d(num_features=16)
        self.prelu1_1 = nn.PReLU(num_parameters=16)

        self.conv2_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(num_features=16)
        self.prelu2_1 = nn.PReLU(num_parameters=16)

        self.conv3_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.bn3_1 = nn.BatchNorm2d(num_features=16)
        self.prelu3_1 = nn.PReLU(num_parameters=16)

        self.conv4_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=1, padding=3)
        self.bn4_1 = nn.BatchNorm2d(num_features=16)
        self.prelu4_1 = nn.PReLU(num_parameters=16)



        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.bn1_2 = nn.BatchNorm2d(num_features=32)
        self.prelu1_2 = nn.PReLU(num_parameters=32)

        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(num_features=32)
        self.prelu2_2 = nn.PReLU(num_parameters=32)

        self.conv3_2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn3_2 = nn.BatchNorm2d(num_features=32)
        self.prelu3_2 = nn.PReLU(num_parameters=32)

        self.conv4_2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.bn4_2 = nn.BatchNorm2d(num_features=32)
        self.prelu4_2 = nn.PReLU(num_parameters=32)



        self.conv1_3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn1_3 = nn.BatchNorm2d(num_features=64)
        self.prelu1_3 = nn.PReLU(num_parameters=64)

        self.conv2_3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2_3 = nn.BatchNorm2d(num_features=64)
        self.prelu2_3 = nn.PReLU(num_parameters=64)

        self.conv3_3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn3_3 = nn.BatchNorm2d(num_features=64)
        self.prelu3_3 = nn.PReLU(num_parameters=64)

        self.conv4_3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.bn4_3 = nn.BatchNorm2d(num_features=64)
        self.prelu4_3 = nn.PReLU(num_parameters=64)

        
        
    def forward(self, input):
        
        op1_1 = self.prelu1_1(self.bn1_1(self.conv1_1(input)))
        op2_1 = self.prelu2_1(self.bn2_1(self.conv2_1(input)))
        op3_1 = self.prelu3_1(self.bn3_1(self.conv3_1(input)))
        op4_1 = self.prelu4_1(self.bn4_1(self.conv4_1(input)))
        op_1 = torch.cat((op1_1, op2_1, op3_1, op4_1), dim=1)

        op1_2 = self.prelu1_2(self.bn1_2(self.conv1_2(op_1)))
        op2_2 = self.prelu2_2(self.bn2_2(self.conv2_2(op_1)))
        op3_2 = self.prelu3_2(self.bn3_2(self.conv3_2(op_1)))
        op4_2 = self.prelu4_2(self.bn4_2(self.conv4_2(op_1)))
        op_2 = torch.cat((op1_2, op2_2, op3_2, op4_2), dim=1)

        op1_3 = self.prelu1_3(self.bn1_3(self.conv1_3(op_2)))
        op2_3 = self.prelu2_3(self.bn2_3(self.conv2_3(op_2)))
        op3_3 = self.prelu3_3(self.bn3_3(self.conv3_3(op_2)))
        op4_3 = self.prelu4_3(self.bn4_3(self.conv4_3(op_2)))
        op_3 = torch.cat((op1_3, op2_3, op3_3, op4_3), dim=1)

        final_logits = torch.mean(torch.sigmoid(op_3.view(opt.batch_size, -1)), 1)
        
        return final_logits