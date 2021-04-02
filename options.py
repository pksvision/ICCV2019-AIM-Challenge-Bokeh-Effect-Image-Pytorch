import argparse
import torch
import os

parser = argparse.ArgumentParser()

parser.add_argument('--no_bokeh_dir', default='/home/prasen/Sujoy/Training/original/')
parser.add_argument('--bokeh_dir', default='/home/prasen/Sujoy/Training/bokeh/')
parser.add_argument('--use_attention', type=bool, default=True)

parser.add_argument('--checkpoints_dir', default='./checkpoints_without_attention/')

parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_images', type=int, default=4692)

parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--channels', type=int, default=1)

parser.add_argument('--learning_rate_g', type=float, default=2e-5)
parser.add_argument('--end_epoch', type=int, default=500)

parser.add_argument('--img_extension', default='.jpg')

parser.add_argument('--beta1', type=float ,default=0.5)
parser.add_argument('--beta2', type=float ,default=0.999)
parser.add_argument('--wd_g', type=float ,default=0.00005)
parser.add_argument('--wd_d', type=float ,default=0.00000)

parser.add_argument('--batch_mse_loss', type=float, default=0.0)
parser.add_argument('--total_mse_loss', type=float, default=0.0)

parser.add_argument('--batch_sobel_loss', type=float, default=0.0)
parser.add_argument('--total_sobel_loss', type=float, default=0.0)
parser.add_argument('--lambda_sobel', type=int, default=0.01)

parser.add_argument('--testing_epoch', type=int, default=1)
parser.add_argument('--testing_mode', default="Syn")

parser.add_argument('--testing_dir_syn', default="./validation/")

opt = parser.parse_args()
print(opt)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(opt.checkpoints_dir):
    os.makedirs(opt.checkpoints_dir)
