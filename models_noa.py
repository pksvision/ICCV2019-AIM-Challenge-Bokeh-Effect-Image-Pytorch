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
from resnet import *

class Bokeh_Generator(nn.Module):

    def __init__(self):
        
        super(Bokeh_Generator, self).__init__()

        self.conv1_l1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1_l1 = nn.BatchNorm2d(num_features=64)
        self.Tanh1_l1 = nn.Tanh()

        self.conv2_l1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2_l1 = nn.BatchNorm2d(num_features=64)
        self.Tanh2_l1 = nn.Tanh()

        self.conv3_l1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3_l1 = nn.BatchNorm2d(num_features=64)
        self.Tanh3_l1 = nn.Tanh()

        self.conv4_l1 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.bn4_l1 = nn.BatchNorm2d(num_features=3)
        self.Tanh4_l1 = nn.Tanh()


        self.conv1_l2 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1_l2 = nn.BatchNorm2d(num_features=64)
        self.Tanh1_l2 = nn.Tanh()

        self.conv2_l2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2_l2 = nn.BatchNorm2d(num_features=64)
        self.Tanh2_l2 = nn.Tanh()

        self.conv3_l2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3_l2 = nn.BatchNorm2d(num_features=64)
        self.Tanh3_l2 = nn.Tanh()

        self.conv4_l2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.bn4_l2 = nn.BatchNorm2d(num_features=3)
        self.Tanh4_l2 = nn.Tanh()


        self.conv1_l3 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1_l3 = nn.BatchNorm2d(num_features=64)
        self.Tanh1_l3 = nn.Tanh()

        self.conv2_l3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2_l3 = nn.BatchNorm2d(num_features=64)
        self.Tanh2_l3 = nn.Tanh()

        self.conv3_l3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3_l3 = nn.BatchNorm2d(num_features=64)
        self.Tanh3_l3 = nn.Tanh()

        self.conv4_l3 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.bn4_l3 = nn.BatchNorm2d(num_features=3)
        self.Tanh4_l3 = nn.Tanh()

        self.resnet = ResNet18()


    def forward(self, input, input_k_sigma, input_k3_sigma, input_k5_sigma, input_canny):


        input_k_sigma = F.pad(input_k_sigma, (2, 2, 2, 2), mode='reflect')
        input_k3_sigma = F.pad(input_k3_sigma, (2, 2, 2, 2), mode='reflect')
        input_k5_sigma = F.pad(input_k5_sigma, (2, 2, 2, 2), mode='reflect')

        input_layer_1 = torch.cat((input_k_sigma, input_canny), dim=1)
        input_layer_2 = torch.cat((input_k3_sigma, input_canny), dim=1)
        input_layer_3 = torch.cat((input_k5_sigma, input_canny), dim=1)

        inp_1_l1 = self.Tanh1_l1(self.bn1_l1(self.conv1_l1(input_layer_1)))
        inp_2_l1 = self.Tanh2_l1(self.bn2_l1(self.conv2_l1(inp_1_l1)))
        inp_3_l1 = self.Tanh3_l1(self.bn3_l1(self.conv3_l1(inp_2_l1)))
        op_l1 = self.Tanh4_l1(self.bn4_l1(self.conv4_l1(inp_3_l1)))

        inp_1_l2 = self.Tanh1_l2(self.bn1_l2(self.conv1_l2(input_layer_2)))
        inp_2_l2 = self.Tanh2_l2(self.bn2_l2(self.conv2_l2(inp_1_l2)))
        inp_3_l2 = self.Tanh3_l2(self.bn3_l2(self.conv3_l2(inp_2_l2)))
        op_l2 = self.Tanh4_l2(self.bn4_l2(self.conv4_l2(inp_3_l2)))

        inp_1_l3 = self.Tanh1_l3(self.bn1_l3(self.conv1_l3(input_layer_3)))
        inp_2_l3 = self.Tanh2_l3(self.bn2_l3(self.conv2_l3(inp_1_l3)))
        inp_3_l3 = self.Tanh3_l3(self.bn3_l3(self.conv3_l3(inp_2_l3)))
        op_l3 = self.Tanh4_l3(self.bn4_l3(self.conv4_l3(inp_3_l3)))

        input_resnet = torch.cat((op_l1, op_l2, op_l3), dim=1)

        output = self.resnet(input_resnet)

        output = output + input

        return output