"""
paper: GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing
file: test.py
about: main entrance for validating/testing the GridDehazeNet
author: Xiaohong Liu
date: 01/08/19
Reference:
@inproceedings{liuICCV2019GridDehazeNet,
    title={GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing},
    author={Liu, Xiaohong and Ma, Yongrui and Shi, Zhihao and Chen, Jun},
    booktitle={ICCV},
    year={2019}
}
"""

# --- Imports --- #
import os
import time
import torch
import argparse
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
from model import GridDehazeNet

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for GridDehazeNet')
parser.add_argument('-network_height', help='Set the network height (row)', default=3, type=int)
parser.add_argument('-network_width', help='Set the network width (column)', default=6, type=int)
parser.add_argument('-num_dense_layer', help='Set the number of dense layer in RDB', default=4, type=int)
parser.add_argument('-growth_rate', help='Set the growth rate in RDB', default=16, type=int)
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-category', help='Set image category (indoor or outdoor?)', default='outdoor', type=str)
parser.add_argument('-input_dir', help='Set the directory of input images', default='./input_images/', type=str)
parser.add_argument('-output_dir', help='Set the directory to save dehazed images', default='./output_images/', type=str)
args = parser.parse_args()

network_height = args.network_height
network_width = args.network_width
num_dense_layer = args.num_dense_layer
growth_rate = args.growth_rate
lambda_loss = args.lambda_loss
val_batch_size = args.val_batch_size
category = args.category
input_dir = args.input_dir
output_dir = args.output_dir

print('--- Hyper-parameters for dehazing ---')
print('val_batch_size: {}\nnetwork_height: {}\nnetwork_width: {}\nnum_dense_layer: {}\ngrowth_rate: {}\nlambda_loss: {}\ncategory: {}'
      .format(val_batch_size, network_height, network_width, num_dense_layer, growth_rate, lambda_loss, category))

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
net = GridDehazeNet(height=network_height, width=network_width, num_dense_layer=num_dense_layer, growth_rate=growth_rate)

# --- Multi-GPU --- #
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)

# --- Load the network weight --- #
net.load_state_dict(torch.load('{}_haze_best_{}_{}'.format(category, network_height, network_width)))

# --- Load input images --- #
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

def load_images(input_dir):
    images = []
    image_names = []
    for file in os.listdir(input_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, file)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            print(f"Input image tensor min/max: {img_tensor.min().item()}, {img_tensor.max().item()}")
            images.append(img_tensor)
            image_names.append(file)
    return images, image_names

images, image_names = load_images(input_dir)

# --- Use the evaluation model in testing --- #
net.eval()
if __name__ == '__main__':
    print('--- Dehazing starts! ---')
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(images):
        with torch.no_grad():
            dehaze = net(img)
        print(f"Dehazed image min/max: {dehaze.min().item()}, {dehaze.max().item()}")
        dehaze = dehaze.squeeze(0).cpu()
        dehaze = dehaze.clamp(0, 1)  # 确保值在 [0, 1] 范围内

        # 转为numpy数组并调整通道顺序
        dehaze = dehaze.numpy().transpose(1, 2, 0)

        # 将像素值缩放到[0, 255]并确保是uint8类型
        dehaze = (dehaze * 255).astype(np.uint8)
        output_path = os.path.join(output_dir, image_names[i])
        Image.fromarray(dehaze).save(output_path, format='png')
        print(f"Dehazed image saved to {output_path}")

    end_time = time.time() - start_time
    print(f"Dehazing time: {end_time:.4f} seconds")