import torch
import torch.nn as nn
import torch.nn.functional as F

from model import UResNet
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from dataset import Love_coco_dataset


def train(model, optimizer, dataloader, loss_fn):
    with tqdm(total = len(dataloader), desc = 'Epoch') as pbar:
        for i, (input_data, ground_truth) in enumerate(dataloader):

            input_data = input_data.cuda()
            ground_truth = ground_truth.cuda()

            optimizer.zero_grad()
            output = model(input_data)[:, 0]
            loss = loss_fn(output, ground_truth)
            loss.backward()
            optimizer.step()
            pbar.set_description(loss = loss.item())
            pbar.update()

            input()


if __name__ == '__main__':
    
    image_path = f'/home/sombra/course/DL/sombra_inpainting/data/img_align_celeba_256/celeba_hq_256'
    synthesis_mask_path = f'/home/sombra/course/DL/sombra_inpainting/data/img_align_celeba_256/synthesis_images'
    dataloader = DataLoader(Love_coco_dataset(image_path, synthesis_mask_path), batch_size = 16, shuffle = True)
    model = UResNet(encoder_depth = 34, num_classes = 1)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    loss_fn = nn.BCEWithLogitsLoss()
    train(model, optimizer, dataloader, loss_fn)