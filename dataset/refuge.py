import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import cv2
from utils import random_box, random_click
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as TF


def display_image_with_point(image, point, box):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.scatter(point[0], point[1], color='red', s=100)  # 点的大小为100

    # 添加边界框
    rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    plt.axis("off")
    plt.show()


def display_labels_with_point(data, point, box):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    titles = ['Label 1', 'Label 2', 'Label 3', 'Label 4']

    for i in range(4):
        axs[i].imshow(data[i], cmap='gray')
        axs[i].scatter(point[0], point[1], color='red', s=100)  # 在每个标签上标记点

        # 在每个子图上添加边界框
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
        axs[i].add_patch(rect)

        axs[i].set_title(titles[i])
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()


class REFUGE(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training'):
        self.data_path = data_path
        self.subfolders = [f.path for f in os.scandir(os.path.join(data_path, mode + '-400')) if f.is_dir()]
        self.mode = mode
        self.prompt = args.prompt
        self.img_size = args.image_size
        self.mask_size = args.out_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.subfolders)

    def __getitem__(self, index):
        point_label = 1

        """Get the images"""
        subfolder = self.subfolders[index]
        name = subfolder.split('/')[-1]

        # raw image and raters path
        img_path = os.path.join(subfolder, name + '.jpg')
        multi_rater_cup_path = [os.path.join(subfolder, name + '_seg_cup_' + str(i) + '.png') for i in range(1, 8)]
        multi_rater_disc_path = [os.path.join(subfolder, name + '_seg_disc_' + str(i) + '.png') for i in range(1, 8)]

        # raw image and raters images
        img = Image.open(img_path).convert('RGB')
        multi_rater_cup = [Image.open(path).convert('L') for path in multi_rater_cup_path]
        multi_rater_disc = [Image.open(path).convert('L') for path in multi_rater_disc_path]

        # resize raters images for generating initial point click
        newsize = (self.img_size, self.img_size)
        multi_rater_cup_np = [np.array(single_rater.resize(newsize)) for single_rater in multi_rater_cup]
        multi_rater_disc_np = [np.array(single_rater.resize(newsize)) for single_rater in multi_rater_disc]

        # first click is the target agreement among most raters
        if 'click' in self.prompt:
            point_label, pt = random_click(np.array(np.mean(np.stack(multi_rater_cup_np), axis=0)) / 255, point_label)
            point_label, pt_disc = random_click(np.array(np.mean(np.stack(multi_rater_disc_np), axis=0)) / 255, point_label)
            x = pt[1]
            y = pt[0]
            pt[0] = x
            pt[1] = y
        else:
            # you may want to get rid of click prompts
            pt = np.array([0, 0], dtype=np.int32)
            
        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            multi_rater_cup = [torch.as_tensor((self.transform(single_rater) >0.5).float(), dtype=torch.float32) for single_rater in multi_rater_cup] #single_rater PILimage transform到512
            multi_rater_cup = torch.stack(multi_rater_cup, dim=0) #7,1,512,512
            # transform to mask size (out_size) for mask define
            mask_cup = F.interpolate(multi_rater_cup, size=(self.mask_size, self.mask_size), mode='bilinear', align_corners=False).mean(dim=0)

            multi_rater_disc = [torch.as_tensor((self.transform(single_rater) >0.5).float(), dtype=torch.float32) for single_rater in multi_rater_disc]
            multi_rater_disc = torch.stack(multi_rater_disc, dim=0)
            mask_disc = F.interpolate(multi_rater_disc, size=(self.mask_size, self.mask_size), mode='bilinear', align_corners=False).mean(dim=0)
            torch.set_rng_state(state)
            
            mask = torch.concat([mask_cup, mask_disc], dim=0)

        if 'box' in self.prompt:
            # print('use box')
            x_min_cup, x_max_cup, y_min_cup, y_max_cup = random_box(multi_rater_cup)  # 7,1,512,512 tensor
            xmin = y_min_cup
            ymin = x_min_cup
            xmax = y_max_cup
            ymax = x_max_cup
            # box_cup = [x_min_cup, x_max_cup, y_min_cup, y_max_cup]
            box_cup = [xmin, ymin, xmax, ymax]
            x_min_disc, x_max_disc, y_min_disc, y_max_disc = random_box(multi_rater_disc) #7,1,512,512 tensor
            box_disc = [x_min_disc, x_max_disc, y_min_disc, y_max_disc]
        else:
            # you may want to get rid of box prompts
            box_cup = [0, 0, 0, 0]
            box_disc = [0, 0, 0, 0]


        #---------------------vis------------------------#
        # display_image_with_point(img.permute(1,2,0),pt,box_cup)
        # conf_angle = mask[:2]
        # angle_width = mask[-2:]
        # combined_slices = torch.cat((conf_angle, angle_width), dim=0)
        # label_resized = F.interpolate(combined_slices.unsqueeze(1),size=(self.img_size,self.img_size),mode="bilinear")
        # display_labels_with_point(label_resized.squeeze(1),pt,box_cup)

        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img, #3,512,512  0-1
            'label': mask, #2,512,512  0-1
            'p_label':point_label, #1
            'pt':pt, #[0,0]
            'box': box_cup, #[0,0,0,0]
            'image_meta_dict':image_meta_dict,
        }