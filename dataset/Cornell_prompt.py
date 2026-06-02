import os
import json
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
from util.data.structure.img import Image
from util.data.structure.grasp import GraspMat, drawGrasp1
import torchvision.transforms.functional as TF
# from .utils_cornell import rescale_bbox, crop, flip_bbox
from utils_cornell import rescale_bbox, crop, flip_bbox
import argparse
from matplotlib.patches import Polygon, Rectangle




def create_star(x_center, y_center, outer_radius, inner_radius):
    star_points = []
    for i in range(10):
        angle = i * np.pi / 5
        radius = outer_radius if i % 2 == 0 else inner_radius
        x = x_center + radius * np.cos(angle)
        y = y_center - radius * np.sin(angle)
        star_points.append((x, y))
    star = Polygon(star_points, closed=True, edgecolor='r', facecolor='yellow')
    return star


# def display_image_with_point(image, point, box):
#     fig, ax = plt.subplots()
#     ax.imshow(image)
#     ax.scatter(point[0], point[1], color='red', s=100)  # 点的大小为100
#
#     # 添加边界框
#     rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
#     ax.add_patch(rect)
#
#     plt.axis("off")
#     plt.show()
#
#
# def display_labels_with_point(data, point, box):
#     fig, axs = plt.subplots(1, 4, figsize=(20, 5))
#     titles = ['Label 1', 'Label 2', 'Label 3', 'Label 4']
#
#     for i in range(4):
#         axs[i].imshow(data[i], cmap='gray')
#         axs[i].scatter(point[0], point[1], color='red', s=100)  # 在每个标签上标记点
#
#         # 在每个子图上添加边界框
#         rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
#         axs[i].add_patch(rect)
#
#         axs[i].set_title(titles[i])
#         axs[i].axis('off')
#
#     plt.tight_layout()
#     plt.show()

def display_image_with_point(image, point, box, save_path='/data1/samgrasp/dataset/test/Cornell_test/show/1.png'):
    fig, ax = plt.subplots()
    ax.imshow(image)
    star = create_star(point[0], point[1], 10, 5)  # 创建五角星
    ax.add_patch(star)

    # 添加边界框
    rect = Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    plt.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()


def display_labels_with_point(data, point, box, save_path='/data1/samgrasp/dataset/test/Cornell_test/show/2.png'):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    titles = ['Label 1', 'Label 2', 'Label 3', 'Label 4']

    for i in range(4):
        axs[i].imshow(data[i], cmap='gray')
        star = create_star(point[0], point[1], 15, 7)  # 创建五角星
        axs[i].add_patch(star)

        rect = Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r',
                         facecolor='none')
        axs[i].add_patch(rect)

        axs[i].set_title(titles[i])
        axs[i].axis('off')

    plt.tight_layout()

    if save_path:
        fig_single, ax_single = plt.subplots(figsize=(5, 5))
        ax_single.imshow(data[0], cmap='gray')
        star = create_star(point[0], point[1], 15, 7)  # 创建五角星
        ax_single.add_patch(star)

        rect = Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r',
                         facecolor='none')
        ax_single.add_patch(rect)

        # ax_single.set_title(titles[0])
        ax_single.axis('off')

        plt.tight_layout()
        fig_single.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig_single)

    plt.show()


class Cornell(Dataset):
    def __init__(self, args, data_path, transform=None, mode = 'Training', bbox = 'gt' #fix
                 ):
        self.data_path = os.path.join(data_path, mode) # /data1/samgrasp/dataset/cornell_adapt
        self.images = []
        self.masks = []


        # 加载所有文件并区分图像和标签
        for filename in os.listdir(self.data_path):
            if filename.endswith(".png"):  # 假设所有图像文件以.png结尾
                if 'mask' in filename:
                    self.masks.append(os.path.join(self.data_path, filename))
                else:
                    self.images.append(os.path.join(self.data_path, filename))

        #-----for cornell only-----#
        self.output_size = 470  # fix 320
        self.angle_k =120
        #--------------------------#
        self.mode = mode
        self.prompt = args.prompt
        self.img_size = args.image_size
        self.mask_size = args.out_size
        self.transform = transform
        self.bbox_mode = bbox
        if self.mode == 'Training':
            self.grounding_dir = 'Training_grounding'
        else:
            self.grounding_dir = 'Test_grounding'


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

    def get_json_path(self, img_path):
        # 获取图像文件名（不带扩展名）
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        # 获取Training文件夹的父目录
        parent_dir = os.path.dirname(os.path.dirname(img_path))
        # 构建Training_grounding文件夹的路径
        grounding_dir = os.path.join(parent_dir, self.grounding_dir)
        # 构建对应的JSON文件路径
        box_json_path = os.path.join(grounding_dir, image_name + '.json')
        return box_json_path

    @staticmethod
    def get_box(box_json_path):
        with open(box_json_path, 'r') as f:
            data = json.load(f)
        # 提取box的值并转换为Tensor
        for item in data:
            if "box" in item:
                box_tensor = torch.tensor(item["box"])
                return box_tensor
        return None

    def __getitem__(self, index):
        point_label = 1

        """Get the images"""

        # raw image and raters path
        img_path = self.images[index] #'/data1/samgrasp/dataset/cornell_adapt/Training/pcd0784r.png'
        label_path = img_path.replace('r.png','grasp.mat') #'/data1/samgrasp/dataset/cornell_adapt/Training/pcd0475grasp.mat'
        box_json_path = self.get_json_path(img_path) #'/data1/samgrasp/dataset/cornell_adapt/Training_grounding/pcd0534r.json'
        img_path = '/data1/samgrasp/dataset/cornell_adapt/Training/pcd0882r.png'
        label_path = '/data1/samgrasp/dataset/cornell_adapt/Training/pcd0882grasp.mat'
        box_json_path = '/data1/samgrasp/dataset/cornell_adapt/Training_grounding/pcd0882r.json'
        # 0108 0487 0514 0882 0784
        # raw image and raters images
        image = Image(img_path) #480,640,3 array
        label = GraspMat(label_path) #4,480,640
        box_json = self.get_box(box_json_path) #tensor 4 topleft , bottomright


        #-----argument------#
        #          resize           #
        scale = np.random.uniform(0.99, 1.01) # fix 0.9  1.1
        image.rescale(scale) #480,640 to 520,694
        label.rescale(scale)
        box_json = rescale_bbox(box_json, scale)



        #           rotate          #
        # rota = 30
        # rota = np.random.uniform(-1 * rota, rota)
        # image.rotate(rota)
        # label.rotate(rota)


        #            crop           #
        dist = 2  # 10 fix
        crop_bbox = image.crop(self.output_size, dist) # to 320,320,3
        label.crop(crop_bbox)
        box_json = crop(box_json, crop_bbox)


        #           flip            #
        # flip = True if np.random.rand() < 0.5 else False
        # if flip:
        #     image.flip()
        #     label.flip()
        #     box_json = flip_bbox(box_json,(self.output_size,self.output_size))



        #            color         #
        # image.color()
        #------------------#

        # img归一化
        image.nomalise()

        img = cv2.cvtColor(image.img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)) #3,320,320
        # 获取target
        label.decode(angle_cls=self.angle_k) #4通道标签变122 4,320,320 to 122,320,320
        target = label.grasp    # (2 + angle_k, 320, 320)

        img = self.numpy_to_torch(img)     #3,320,320 under1
        target = self.numpy_to_torch(target)   #122,320,320 tensor 0 or 1


        # resize raters images for generating initial point click
        newsize = (self.img_size, self.img_size) #512,512
        click_mask_np = cv2.resize(target[0].numpy().copy(), newsize, interpolation = cv2.INTER_LINEAR)

        if 'click' in self.prompt:
            point_label, pt = random_click(click_mask_np , point_label) #in: inputsize,input 0-1
            x = pt[1]
            y = pt[0]
            pt[0] = x
            pt[1] = y
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
            img = self.transform(TF.to_pil_image(img))  #3,320,320 to 3 512 512tensor to input size  0-1  ------------#-----------

            # transform to mask size (out_size) for mask define
            # conf_mask = (self.transform(target[:1]) > 0.5).float()
            conf_mask = F.interpolate(target[:1].unsqueeze(0), size=(self.mask_size, self.mask_size), mode='bilinear',
                                     align_corners=False)  #

            angle_mask = F.interpolate(target[1:-1].unsqueeze(0), size=(self.mask_size, self.mask_size), mode='bilinear',
                                     align_corners=False) #

            width_mask = F.interpolate(target[-1:].unsqueeze(0), size=(self.mask_size, self.mask_size), mode='bilinear',
                                      align_corners=False) #1,1,512,512
            torch.set_rng_state(state)

            mask = torch.concat([conf_mask.squeeze(0)
                                    ,angle_mask.squeeze(0)
                                    , width_mask.squeeze(0)
                                 ], dim=0) #122,512,512



        if 'box' in self.prompt:
            if self.bbox_mode == 'gt':
                box_mask = F.interpolate(conf_mask, size=(self.img_size, self.img_size), mode="bilinear")
                box_mask = torch.as_tensor(box_mask > 0.5, dtype=torch.float32)
                x_min_cup, x_max_cup, y_min_cup, y_max_cup = random_box(box_mask) #ymin,ymax,xmin,xmax
                xmin = y_min_cup
                ymin = x_min_cup
                xmax = y_max_cup
                ymax = x_max_cup
                # box_cup = [x_min_cup, x_max_cup, y_min_cup, y_max_cup]
                box_cup = [xmin, ymin, xmax, ymax]
            elif self.bbox_mode == 'gd':
                box_json = rescale_bbox(box_json, self.img_size / self.output_size )
                box_cup = box_json.tolist()


            # x_min_disc, x_max_disc, y_min_disc, y_max_disc = random_box(box_mask)
            # box_disc = [x_min_disc, x_max_disc, y_min_disc, y_max_disc]
        else:
            # you may want to get rid of box prompts
            box_cup = [0, 0, 0, 0]
            box_disc = [0, 0, 0, 0]

        #------visualize-------#
        display_image_with_point(img.permute(1,2,0),pt,box_cup)
        conf_angle = mask[:2]
        angle_width = mask[-2:]
        combined_slices = torch.cat((conf_angle, angle_width), dim=0)
        label_resized = F.interpolate(combined_slices.unsqueeze(1),size=(self.img_size,self.img_size),mode="bilinear")
        display_labels_with_point(label_resized.squeeze(1),pt,box_cup)
        #----------------------#
        img_path #0108 0487 0514 0882 0784
        pass

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
        parser.add_argument('-crop_size', type=str, default=500, help='out size')
        parser.add_argument('-out_size', type=str, default=256, help='out size')
        parser.add_argument('-data_path', type=str, default='/data1/samgrasp/dataset/cornell_adapt', help='data_path')
        opt = parser.parse_args()
        return opt
    args = parse_args()
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
    Cornell_Dataset = Cornell(args, args.data_path, transform=transform_train,
                                         mode='Training')
    nice_train_loader = torch.utils.data.DataLoader(Cornell_Dataset, batch_size=1, shuffle=True, num_workers=1,
                                   pin_memory=True)

    print('>> dataset: {}'.format(len(nice_train_loader)))

    count = 0
    max_w = 0
    for det in nice_train_loader:
        print('loaded')
        pass