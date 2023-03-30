import cv2
import matplotlib.pyplot as plt
import glob
import random
from PIL import Image
from utils import log
from tqdm import trange
target_folder = f'/home/sombra/course/DL/sombra_inpainting/data/img_align_celeba_256/synthesis_images'

image_folder = f'/home/sombra/course/DL/sombra_inpainting/data/img_align_celeba_256/celeba_hq_256'
# mask_folder = [f'/home/sombra/course/DL/sombra_inpainting/data/qd_img/qd_imd/test',
#                f'/home/sombra/course/DL/sombra_inpainting/data/qd_img/qd_imd/train']
mask_folder = f'/home/sombra/course/DL/sombra_inpainting/data/qd_img/qd_imd'

# mask_file_list = glob.glob(mask_folder + '/*.jpg')
image_file_list = glob.glob(f'{image_folder}/*.jpg')
mask_file_list = glob.glob(f'{mask_folder}/**/*.png', recursive=True)
log(f"image_num: {len(image_file_list)}, mask_num: {len(mask_file_list)}")

for i in trange(len(image_file_list)):
    image_path = image_file_list[i]
    img_name = image_path.split('/')[-1].split('.')[0] + '.png'

    mask_idx = random.randint(0, len(mask_file_list) - 1)
    mask_path = mask_file_list[mask_idx]
    mask_name = mask_path.split('/')[-1].split('.')[0] + '.png'
    # log(f"{img_name}, {mask_name}") 

    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path)
    mask = cv2.resize(mask, (image.shape[0], image.shape[1]), interpolation = cv2.INTER_AREA)    
    mask[mask < 128] = 0
    mask[mask >= 128] = 255
    # log(f"image.shape: {image.shape}, mask.shape: {mask.shape}")
    assert (mask[:, :, 1] == mask[:, :, 2]).all() and (mask[:, :, 0] == mask[:, :, 1]).all()
    mask = mask[:, :, 1]
    
    synthesis_image = image.copy()
    synthesis_image[mask == 0] = 0

    target_file_path = f'{target_folder}/{img_name}'
    cv2.imwrite(target_file_path, cv2.cvtColor(synthesis_image, cv2.COLOR_RGB2BGR))

    test_image = cv2.cvtColor(cv2.imread(target_file_path), cv2.COLOR_BGR2RGB)
    assert (test_image == synthesis_image).all() == True

    # log(f"{(test_image == synthesis_image).all()}")
    # plt.figure('image')
    # plt.imshow(image)
    # plt.figure('mask')
    # plt.imshow(mask, cmap='gray')
    # plt.figure('synthesis_image')
    # plt.imshow(synthesis_image)
    # plt.show()
