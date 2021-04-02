import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import cv2
import os
import numpy as np
from options import opt
import torchvision.transforms.functional as F
import numbers
import random
from PIL import Image
from torchvision import transforms as transfm
import random
# from net import *

class ToTensor(object):

    def __call__(self, sample):
        
        no_bokeh_image, bokeh_image = np.array(sample['no_bokeh']), np.array(sample['bokeh'])
                
        no_bokeh_image = no_bokeh_image.astype(np.float32)
        no_bokeh_image = torch.from_numpy(no_bokeh_image).unsqueeze(2)
        no_bokeh_image = torch.transpose(torch.transpose(no_bokeh_image, 2, 0), 1, 2)
        no_bokeh_image = (2 * no_bokeh_image / 255) - 1

        bokeh_image = bokeh_image.astype(np.float32)
        bokeh_image = torch.from_numpy(bokeh_image).unsqueeze(2)
        bokeh_image = torch.transpose(torch.transpose(bokeh_image, 2, 0), 1, 2)
        bokeh_image = (2 * bokeh_image / 255) - 1 

        return {'no_bokeh': no_bokeh_image,
                'bokeh': bokeh_image}

class Dataset_Load(Dataset):

    def __init__(self, no_bokeh_path, bokeh_path, transform=None):

        self.no_bokeh_dir = no_bokeh_path
        self.bokeh_dir = bokeh_path
        self.transform = transform

    def __len__(self):

        return opt.num_images

    def apply_transforms(self, image, output):
        if random.random() > 0.5:
            image = F.hflip(image)
            output = F.hflip(output)

        if random.random() > 0.5:
            image = F.vflip(image)
            output = F.vflip(output)

        return image, output

    def __getitem__(self, index):
        
        no_boekeh_image_name = str(index) + opt.img_extension
        no_bokeh_im = cv2.resize(cv2.imread(os.path.join(self.no_bokeh_dir, no_boekeh_image_name)), (256, 256), interpolation=cv2.INTER_CUBIC)
        no_bokeh_im = cv2.cvtColor(no_bokeh_im, cv2.COLOR_BGR2YCR_CB)
        no_bokeh_im_y, _, _ = cv2.split(no_bokeh_im)

        bokeh_image_name = str(index) + opt.img_extension
        bokeh_im = cv2.resize(cv2.imread(os.path.join(self.bokeh_dir, bokeh_image_name)), (256, 256), interpolation=cv2.INTER_CUBIC)
        bokeh_im = cv2.cvtColor(bokeh_im, cv2.COLOR_BGR2YCR_CB)
        bokeh_im_y, _, _ = cv2.split(bokeh_im)

        bokeh_im_y_pil = Image.fromarray(bokeh_im_y)
        no_bokeh_im_y_pil = Image.fromarray(no_bokeh_im_y)
        transform_list = []
        transform_list.append(transfm.RandomRotation(20))
        transform_list.append(transfm.RandomHorizontalFlip(0.5))
        transform_list.append(transfm.RandomVerticalFlip(0.25))
        final_transform = transfm.Compose(transform_list)

        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        no_bokeh_im_y = final_transform(no_bokeh_im_y_pil)

        random.seed(seed) # apply this seed to img tranfsorms
        bokeh_im_y = final_transform(bokeh_im_y_pil)

        # no_bokeh_im_y, bokeh_im_y = self.apply_transforms(no_bokeh_im_y_pil, bokeh_im_y_pil)

        sample = {'no_bokeh': no_bokeh_im_y, 
                  'bokeh': bokeh_im_y}

        if self.transform != None:
            sample = self.transform(sample)

        return sample