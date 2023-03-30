import torch
import torch.nn as nn
import torch.nn.functional as F

from UResnet import UResnet
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from dataset import Love_coco_dataset


def train(model, optimizer, dataloader, loss_fn):
    with tqdm(total = len(dataloader), desc = 'Epoch') as pbar:
        for i, (input_data, ground_truth) in enumerate(dataloader):

            input_data = input_data.cuda()
            ground_truth = ground_truth.cuda()

            optimizer.zero_grad()
            output = model(input_data)
            loss = loss_fn(output, ground_truth)
            loss.backward()
            optimizer.step()
            pbar.set_description(loss = loss.item())
            pbar.update()


if __name__ == '__main__'：
    image_path = f'/home/sombra/course/DL/sombra_inpainting/data/img_align_celeba_256/celeba_hq_256'
    synthesis_mask_path = f'/home/sombra/course/DL/sombra_inpainting/data/img_align_celeba_256/synthesis_images'
    dataloader = DataLoader(Love_coco_dataset(image_path, synthesis_mask_path), batch_size = 16, shuffle = True)
    model = UResnet()
    model.cuda()
    optimizer = nn.optim.Adam(model.parameters(), lr = 0.001)
    loss_fn = nn.BCEWithLogitsLoss()
    train(model, dataloader, loss_fn)