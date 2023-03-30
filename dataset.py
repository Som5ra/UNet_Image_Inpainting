from torch.utils.data import Dataset
import pandas as pd
import os
import cv2
from PIL import Image
from utils import log
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import numpy as np

class Love_coco_dataset(Dataset):   
    def __init__(self, image_path, synthesis_mask_path, image_type='.jpg', mask_type='.png'):
        super().__init__()
        self.img_dir = image_path
        self.image_list = os.listdir(image_path)
        self.syn_path = synthesis_mask_path
        self.syn_image_list = os.listdir(synthesis_mask_path)
        self.image_type = image_type
        self.mask_type = mask_type
        self.image_mask_pairs = self.load_data()

    def __getitem__(self, idx):

        image = self.image_mask_pairs[idx][0]
        syn_image = self.image_mask_pairs[idx][1]
        # plt.figure('image')
        # plt.imshow(image)
        # plt.figure('syn_image')
        # plt.imshow(syn_image)
        # plt.show()
        image = torch.from_numpy(image).float().div(255).cuda()
        syn_image = torch.from_numpy(syn_image).float().div(255).cuda()
        image = image.permute(2, 0, 1)
        syn_image = syn_image.permute(2, 0, 1)
        return image, syn_image
    
    def __len__(self):
        return len(self.image_mask_pairs)

    def load_data(self):
        pairs = []
        for i in tqdm(range(len(self.image_list[:20]))):
            true_idx = self.image_list[i].split('.')[0]
            image = np.ascontiguousarray(cv2.imread(os.path.join(self.img_dir, true_idx + self.image_type))[:, :, ::-1])
            syn_image = np.ascontiguousarray(cv2.imread(os.path.join(self.syn_path, true_idx + self.mask_type))[:, :, ::-1])
            pairs.append((np.ascontiguousarray(image), syn_image))

        return pairs
    
if __name__ == '__main__':
    img_dir = f'/home/sombra/course/DL/sombra_inpainting/data/img_align_celeba_256/celeba_hq_256'
    syn_path = f'/home/sombra/course/DL/sombra_inpainting/data/img_align_celeba_256/synthesis_images'

    love_you_dataset = Love_coco_dataset(img_dir, syn_path)
    for i, x in enumerate(love_you_dataset):
        continue

