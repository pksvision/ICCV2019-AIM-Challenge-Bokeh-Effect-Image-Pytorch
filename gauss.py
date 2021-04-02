import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F
import cv2
import numpy as np

class GaussianSmoothing(nn.Module):

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
    
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        kernel = kernel / torch.sum(kernel)

        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel.float())
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        
        return self.conv(input.float(), weight=self.weight, groups=self.groups)


# if __name__=="__main__":

#     k = 2**(1/2)
#     sigma = 1.6
#     channels = 3
#     gaussian_kernel = 5

#     img, _, _ = cv2.split(cv2.cvtColor(cv2.imread("./gaussians/sample.jpg"), cv2.COLOR_BGR2YCR_CB))
#     img = img/255.0

#     img = torch.transpose(torch.transpose(torch.from_numpy(img).unsqueeze(2), 1, 2), 0 ,1)
#     img = img.unsqueeze(0)

#     img = torch.cat((img, img, img), dim=1)
#     img = img.cuda()
#     print("Input shape : ", img.shape)

#     print("sigma = ",sigma)
#     smoothing = GaussianSmoothing(channels, gaussian_kernel, sigma).cuda()
#     input = F.pad(img, (2, 2, 2, 2), mode='reflect')
#     output = smoothing(input)
#     final = torch.transpose(torch.transpose(output[0], 0, 1),1,2)
#     cv2.imwrite("./gaussians/sample_gaussian_sigma.png", np.uint8(final.cpu().detach().numpy()*255.0))
#     print(np.uint8(final.cpu().detach().numpy()*255.0).shape)


#     print("sigma = ", (k**(5))*sigma)
#     smoothing = GaussianSmoothing(channels, gaussian_kernel, (k**(5))*sigma).cuda()
#     input = F.pad(img, (2, 2, 2, 2), mode='reflect')
#     output = smoothing(input)
#     final = torch.transpose(torch.transpose(output[0], 0, 1),1,2)
#     cv2.imwrite("./gaussians/sample_gaussian_K5sigma.png", np.uint8(final.cpu().detach().numpy()*255.0))
#     print(np.uint8(final.cpu().detach().numpy()*255.0).shape)


#     print("sigma = ", (k**(14))*sigma)
#     smoothing = GaussianSmoothing(channels, gaussian_kernel, (k**(14))*sigma).cuda()
#     input = F.pad(img, (2, 2, 2, 2), mode='reflect')
#     output = smoothing(input)
#     final = torch.transpose(torch.transpose(output[0], 0, 1),1,2)
#     cv2.imwrite("./gaussians/sample_gaussian_K14sigma.png", np.uint8(final.cpu().detach().numpy()*255.0))
#     print(np.uint8(final.cpu().detach().numpy()*255.0).shape)

