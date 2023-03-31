import torch
import torch.nn as nn
import torch.nn.functional as F

from model import UResNet
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from dataset import Love_coco_dataset
from loss import build_vgg_loss, build_CE_loss, VGG16
from utils import load_model, save_model
from torch.utils.tensorboard import SummaryWriter

def train(model, optimizer, dataloader, epochs, vis = False):

    if vis:
        writer = SummaryWriter(log_dir = '/home/sombra/course/DL/sombra_inpainting/weights/tensorboard_logs')

    Vgg_for_loss = VGG16()
    Vgg_for_loss.cuda()

    for epoch in range(1, epochs + 1):
        with tqdm(total = len(dataloader), desc = 'Epoch') as pbar:
            pbar.set_description(f'Epoch = {epoch}')
            tot_loss = 0.
            for i, (input_data, ground_truth) in enumerate(dataloader):
                input_data = input_data.cuda()
                ground_truth = ground_truth.cuda()
                optimizer.zero_grad()
                output = model(input_data)

                vgg_loss = build_vgg_loss(Vgg_for_loss, torch.cat([output, ground_truth], dim = 0))
                l_per, l_style = vgg_loss
                ce_loss = build_CE_loss(output, ground_truth)
                loss = l_per + l_style + ce_loss

                if vis:
                    writer.add_scalar("CE Loss", ce_loss, epoch)
                    writer.add_scalar("Perceptual Loss", l_per, epoch)
                    writer.add_scalar("Style Loss", l_style, epoch)
                    writer.add_scalar("Loss", loss, epoch)

                tot_loss += loss.item()
                loss.backward()
                optimizer.step()
                pbar.set_postfix(f'loss = {tot_loss / (i + 1)}', refresh = True)
                pbar.update()

            if vis:
                writer.flush()

            save_model(model, optimizer, epoch, loss, f'/home/sombra/course/DL/sombra_inpainting/weights/latest_epoch.pth')
            if epoch % 10 == 0:
                save_model(model, optimizer, epoch, loss, f'/home/sombra/course/DL/sombra_inpainting/weights/epoch_{epoch}.pth')
    if vis:
        writer.close()

def coco_train(epochs = 100, vis = True):
    image_path = f'/home/sombra/course/DL/sombra_inpainting/data/img_align_celeba_256/celeba_hq_256'
    synthesis_mask_path = f'/home/sombra/course/DL/sombra_inpainting/data/img_align_celeba_256/synthesis_images'
    dataloader = DataLoader(Love_coco_dataset(image_path, synthesis_mask_path), batch_size = 32, shuffle = True)

    model = UResNet(encoder_depth = 34, output_channels=3, num_filters=32, dropout_2d=0.05, pretrained=True)
    model.train()
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    train(model, optimizer, dataloader, epochs = epochs, vis = vis)

if __name__ == '__main__':
    coco_train(epochs = 100, vis = True)