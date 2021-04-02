import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class Self_Attn(nn.Module):

    def __init__(self,in_dim,activation):
        
        super(Self_Attn,self).__init__()
        
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = spectral_norm(nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1))
        self.key_conv = spectral_norm(nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1))
        self.value_conv = spectral_norm(nn.Conv2d(in_channels = in_dim , out_channels = in_dim//2 , kernel_size= 1))
        self.atten_conv = spectral_norm(nn.Conv2d(in_channels=in_dim//2, out_channels=in_dim, kernel_size=1))
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1) 

        self.upscale = nn.Upsample(scale_factor=4, mode='nearest')
        self.downscale = nn.MaxPool2d(3, stride=4)


    def forward(self,x):

        x_bar = self.downscale(x)
        
        m_batchsize,C,width ,height = x_bar.size()
        proj_query  = self.query_conv(x_bar).view(m_batchsize,-1,width*height).permute(0,2,1) 
        proj_key =  self.key_conv(x_bar).view(m_batchsize,-1,width*height) 
        energy =  torch.bmm(proj_query,proj_key) 
        attention = self.softmax(energy) 
        proj_value = self.value_conv(x_bar).view(m_batchsize,-1,width*height)
        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C//2,width,height)
        out = self.atten_conv(out)

        out = self.upscale(out)
        print ("Attention")
        final = self.gamma*out + x
    
        return final