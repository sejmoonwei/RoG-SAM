import os
import glob
import argparse

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
from util.data.structure.img_jacquard import Image
from util.data.structure.grasp_jacquard import GraspMat, drawGrasp1
import torchvision.transforms.functional as TF


def display_image_with_point(image, point, box):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.scatter(point[0], point[1], color='red', s=10)  # 点的大小为100

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



class Jacquard(Dataset):
    def __init__(self, args, data_path, transform=None, mode='Training',
                 plane=False):
        self.images = []
        self.masks = []
        # 加载所有文件并区分图像和标签
        if mode == 'Training':
            image_file = glob.glob(os.path.join(data_path, '*', '*', '*_RGB.png'))
        else:
            image_file = glob.glob(os.path.join('/data1/samgrasp/dataset/test/jaq_test', '*', '*', '*_RGB.png'))
        self.images = [file for file in image_file]
        self.masks = [file.replace('_RGB.png','_grasps.txt') for file in image_file]


        #-----for Jacquard only-----#
        self.output_size = 500
        self.angle_k =120
        #--------------------------#
        self.mode = mode
        self.prompt = args.prompt
        self.img_size = args.image_size
        self.mask_size = args.out_size
        self.transform = transform


    def __len__(self):
        return len(self.images)

    @staticmethod
    def numpy_to_torch(s):
        """
        numpy转tensor
        """
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def __getitem__(self, index):
        point_label = 1
        """Get the images"""

        # raw image and label path
        img_path = self.images[index]
        # print(img_path)
        label_path = img_path.replace('RGB.png','grasps.txt')
        # raw image and raters images
        image = Image(img_path)
        label = GraspMat(label_path)

        #-----argument------#
        # resize
        # scale = np.random.uniform(0.98, 1.02)
        # image.rescale(scale)
        # label.rescale(scale)
        # rotate
        rota = 30
        rota = np.random.uniform(-1 * rota, rota)
        image.rotate(rota)
        label.rotate(rota)
        # crop
        dist = 2  # 50
        crop_bbox = image.crop(self.output_size, dist)
        label.crop(crop_bbox)
        # flip
        flip = True if np.random.rand() < 0.5 else False
        if flip:
            image.flip()
            label.flip()
        # color
        image.color()
        #------------------#

        # img归一化
        image.nomalise()

        img = cv2.cvtColor(image.img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1))
        # 获取target
        label.decode(angle_cls=self.angle_k) #4通道标签变122
        target = label.grasp    # (2 + angle_k, 320, 320)

        img = self.numpy_to_torch(img)     #3,320,320 under1
        target = self.numpy_to_torch(target)   #122,400,400 tensor 0 or 1


        # resize raters images for generating initial point click
        newsize = (self.img_size, self.img_size)
        click_mask_np = cv2.resize(target[0].numpy().copy(), newsize, interpolation = cv2.INTER_NEAREST)

        if 'click' in self.prompt:
            point_label, pt = random_click(click_mask_np , point_label) #in: inputsize,input 0-1
            x = pt[1]
            y = pt[0]
            pt[0] = x
            pt[1] = y
            # point_label, pt_disc = random_click(np.array(target[0]) , point_label)
        else:
            # you may want to get rid of click prompts
            pt = np.array([0, 0], dtype=np.int32)


        # first click is the target agreement among most raters


        if self.transform:
            state = torch.get_rng_state()
            #---------------nomalize----------#
            # 假设 img 是一个形状为 [3, 320, 320]，值范围为 -0.6 到 0.18 的张量
            min_val = img.min()
            max_val = img.max()
            # 将图像数据归一化到 [0, 1]
            img = (img - min_val) / (max_val - min_val)
            #---------------------------------#
            img = self.transform(TF.to_pil_image(img))  #3,320,320 tensor to input size  0-1  ------------#-----------

            # transform to mask size (out_size) for mask define
            # conf_mask = (self.transform(target[:1]) > 0.5).float()
            conf_mask = F.interpolate(target[:1].unsqueeze(0), size=(self.mask_size, self.mask_size), mode='nearest',
                                     )  #

            angle_mask = F.interpolate(target[1:-1].unsqueeze(0), size=(self.mask_size, self.mask_size), mode='nearest',
                                     ) #

            width_mask = F.interpolate(target[-1:].unsqueeze(0), size=(self.mask_size, self.mask_size), mode='nearest',
                                      ) #1,1,512,512   for angle and width , not sure to use bilinear or nearest
            torch.set_rng_state(state)

            mask = torch.concat([conf_mask.squeeze(0)
                                    ,angle_mask.squeeze(0)
                                    , width_mask.squeeze(0)
                                 ], dim=0) #122,512,512



        if 'box' in self.prompt:
            box_mask = F.interpolate(conf_mask, size=(self.img_size, self.img_size), mode="nearest")
            box_mask = torch.as_tensor(box_mask > 0.0, dtype=torch.float32)
            x_min_cup, x_max_cup, y_min_cup, y_max_cup = random_box(box_mask) #ymin,ymax,xmin,xmax
            xmin = y_min_cup
            ymin = x_min_cup
            xmax = y_max_cup
            ymax = x_max_cup
            # box_cup = [x_min_cup, x_max_cup, y_min_cup, y_max_cup]
            box_cup = [xmin, ymin, xmax, ymax]


            # x_min_disc, x_max_disc, y_min_disc, y_max_disc = random_box(box_mask)
            # box_disc = [x_min_disc, x_max_disc, y_min_disc, y_max_disc]
        else:
            # you may want to get rid of box prompts
            box_cup = [0, 0, 0, 0]
            box_disc = [0, 0, 0, 0]

        #------visualize-------#
        # display_image_with_point(img.permute(1,2,0),pt,box_cup)
        # conf_angle = mask[:2]
        # angle_width = mask[-2:]
        # combined_slices = torch.cat((conf_angle, angle_width), dim=0)
        # label_resized = F.interpolate(combined_slices.unsqueeze(1),size=(self.img_size,self.img_size),mode="nearest")
        # display_labels_with_point(label_resized.squeeze(1),pt,box_cup)
        #----------------------#

        image_meta_dict = {'filename_or_obj': img_path.split('/')[-1]}
        return {
            'image': img,
            'label': mask, #0-1
            'p_label': point_label,
            'pt': pt,
            'box': box_cup,
            'image_meta_dict': image_meta_dict,
        }

if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('-prompt', nargs='+', type=str, default=['click','box'], help='Enter one or more valid options.')
        parser.add_argument('-image_size', type=str, default=512, help='input size')
        parser.add_argument('-out_size', type=str, default=256, help='out size')

        opt = parser.parse_args()
        return opt
    args = parse_args()
    angle_cls = 120
    data_path = '/data1/samgrasp/dataset/test/jaq_test'
    # 加载训练集
    print('Loading Dataset...')
    import torchvision.transforms as transforms
    transform_train = transforms.Compose([
        transforms.Resize((args.image_size,args.image_size)),
        transforms.ToTensor(),
    ])
    transform_train_seg = transforms.Compose([
        transforms.Resize((args.out_size,args.out_size)),
        transforms.ToTensor(),
    ])
    Jacquard_train_dataset = Jacquard(args, data_path, transform=transform_train,
                                     mode='Training')
    train_data = torch.utils.data.DataLoader(
        Jacquard_train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1)

    print('>> dataset: {}'.format(len(train_data)))

    count = 0
    max_w = 0
    for det in train_data:
        pass
        count += 1
        x = det['image']
        y = det['label']
        img = x[0].cpu().numpy().transpose((1, 2, 0)).astype(np.uint8)  # (3, h, w) -> (h, w, 3)
        im = np.zeros(img.shape, dtype=np.uint8)
        im[:, :, 0] = img[:, :, 0]
        im[:, :, 1] = img[:, :, 1]
        im[:, :, 2] = img[:, :, 2]

        able = y[0, 0, :, :].cpu().numpy()  # (320, 320)
        angles = y[0, 1:-1, :, :].cpu().numpy()  # (angle_k, 320, 320)
        widths = y[0, -1, :, :].cpu().numpy()  # (320, 320)

        rows, cols = np.where(able == 1)  # 抓取点
        for i in range(rows.shape[0]):
            row, col = rows[i], cols[i]
            width = widths[row, col] * 100
            angle = np.argmax(angles[:, row, col]) / angle_cls * 2 * np.pi
            drawGrasp1(im, [row, col, angle, width])

        cv2.imwrite('/home/wangdx/research/grasp_detection/img/test/' + str(count) + '.png', im)