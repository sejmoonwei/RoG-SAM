import numpy as np
import cv2
from utils.data import get_dataset
import torch
import torch.utils.data
import math
import random
import os
import copy
import matplotlib.pyplot as plt

from utils.data.structure.img import Image
from utils.data.structure.grasp import GraspMat, drawGrasp1

def display_image(image):
     plt.imshow(image)
     plt.axis("off")
     plt.show()

def display_labels(data):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    titles = ['Label 1', 'Label 2', 'Label 3', 'Label 4']

    for i in range(4):
        axs[i].imshow(data[i], cmap='gray')
        axs[i].set_title(titles[i])
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()




class GraspDatasetBase(torch.utils.data.Dataset):
    def __init__(self, output_size, angle_k, include_depth=False, include_rgb=True,
                 argument=False):
        """
        """

        self.output_size = output_size
        self.include_depth = include_depth
        self.include_rgb = include_rgb
        self.angle_k = angle_k
        self.argument = argument

        if include_depth is False and include_rgb is False:
            raise ValueError('At least one of Depth or RGB must be specified.')

    @staticmethod
    def numpy_to_torch(s):
        """
        """
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def __getitem__(self, idx):
        label_name = self.grasp_files[idx]
        rgb_name = label_name.replace('grasp.mat', 'r.png')

        image = Image(rgb_name)
        label = GraspMat(label_name)
        if self.argument:
            # resize
            scale = np.random.uniform(0.9, 1.1)
            image.rescale(scale)
            label.rescale(scale)
            # rotate
            rota = 30
            rota = np.random.uniform(-1 * rota, rota)
            image.rotate(rota)
            label.rotate(rota)
            # crop
            dist = 30   # 50
            crop_bbox = image.crop(self.output_size, dist)
            label.crop(crop_bbox)
            # flip
            flip = True if np.random.rand() < 0.5 else False
            if flip:
                image.flip()
                label.flip()
            # color
            image.color()
        else:
            # crop
            crop_bbox = image.crop(self.output_size)
            label.crop(crop_bbox)

        #visualize
        # display_image(image.img)
        # display_labels(label.grasp)


        image.nomalise()
        img = image.img.transpose((2, 0, 1))  # (320, 320, 3) -> (3, 320, 320)
        label.decode(angle_cls=self.angle_k)
        target = label.grasp    # (2 + angle_k, 320, 320)

        img = self.numpy_to_torch(img)
        target = self.numpy_to_torch(target)

        return img, target


    def __len__(self):
        return len(self.grasp_files)
