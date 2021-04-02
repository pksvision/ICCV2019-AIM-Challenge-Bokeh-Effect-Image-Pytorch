import torch
import torch.nn as nn
import cv2
import numpy as np
import os

def get_sobel_kernel_x():

    K_X= torch.tensor([
                        [-1., 0., 1.],
                        [-2., 0., 2.],
                        [-1., 0., 1.],
                        ])
    K_Y = K_X.t()
    K_X = torch.unsqueeze(torch.unsqueeze(K_X,0),0)
    K_Y = torch.unsqueeze(torch.unsqueeze(K_Y,0),0)
    # K_X = torch.cat((K_X, K_X, K_X), dim=1)
    # K_Y = torch.cat((K_Y, K_Y, K_Y), dim=1)
    return [K_X, K_Y]


class Sobel_Op(nn.Module):

    def __init__(self):
        
        super(Sobel_Op,self).__init__()

        self.grad_x = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.grad_y = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)

        with torch.no_grad():
            self.grad_x.weight = torch.nn.Parameter(get_sobel_kernel_x()[0], requires_grad=False)
            self.grad_y.weight = torch.nn.Parameter(get_sobel_kernel_x()[1], requires_grad=False)

            self.grad_x.bias = None
            self.grad_y.bias = None


    def forward(self,input):

        sobel_x = self.grad_x(input)
        sobel_y = self.grad_y(input)

        return [sobel_x, sobel_y]


# if __name__=='__main__':

#     sobel_net = Sobel_Op()

#     dirs="./Sobels"
#     image = "no_rain.jpg"
#     image_name = image.split(".")[0]

#     im = cv2.imread(os.path.join(dirs, image))
#     im = im/255.0
#     im = im.astype(np.float32)
#     im = torch.from_numpy(im)
#     im = torch.transpose(torch.transpose(im, 2, 0), 1, 2)
#     im = im.unsqueeze(0)

#     print(im.shape)
    
#     sobels = sobel_net(im)

#     # print(sobels[0].shape, sobels[1].shape)
    
#     sobels[0] = torch.transpose(torch.transpose(sobels[0], 1 ,2), 2, 3)
#     sobels[0] = np.uint8(sobels[0].detach().numpy())    
    
#     cv2.imwrite(os.path.join(dirs, image_name+'_x.png'),sobels[0][0])


#     sobels[1] = torch.transpose(torch.transpose(sobels[1], 1 ,2), 2, 3)
#     sobels[1] = np.uint8(sobels[1].detach().numpy())    
    
#     cv2.imwrite(os.path.join(dirs, image_name+'_y.png'),sobels[1][0])