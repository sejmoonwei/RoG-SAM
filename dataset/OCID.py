import os
import glob
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
# from PIL import Image
from torch.utils.data import Dataset
import cv2
from utils import random_box, random_click
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from util.data.structure.img_OCID import Image
from util.data.structure.grasp_OCID import GraspMat, drawGrasp1
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

class OCID(Dataset):
    """OCID_grasp dataset for grasp detection and semantic segmentation
    """
    def __init__(self,args,  transform,  split_name , data_path ):
        super(OCID, self).__init__()
        self.angle_k = 120
        self.data_path = data_path
        self.split_name = split_name
        self.transform = transform
        self.img_size = args.image_size
        self.crop_size = 500
        self.mask_size = args.out_size
        self._images = self._load_split()
        self.prompt = args.prompt
        self.mode = 'subset'


    def _load_split(self):
        with open(os.path.join(self.data_path, self.split_name + ".txt"), "r") as fid:
            images = [x.strip() for x in fid.readlines()]
        return images

    def _get_path(self, item):
        seq_path, im_name = item.split(',')
        sample_path = os.path.join(self.data_path, seq_path)
        img_path = os.path.join(sample_path, 'rgb', im_name)
        ins_mask_path = os.path.join(sample_path, 'seg_mask_instances_combi', im_name)
        anno_path = os.path.join(sample_path, 'Annotations', im_name[:-4] + '.txt')
        return img_path, anno_path, ins_mask_path

    @property
    def categories(self):
        """Category names"""
        return self._meta["categories"]

    @property
    def num_categories(self):
        """Number of categories"""
        return len(self.categories)

    @property
    def num_stuff(self):
        """Number of "stuff" categories"""
        return self._meta["num_stuff"]

    @property
    def num_thing(self):
        """Number of "thing" categories"""
        return self.num_categories - self.num_stuff

    @property
    def original_ids(self):
        """Original class id of each category"""
        return self._meta["original_ids"]

    @property
    def palette(self):
        """Default palette to be used when color-coding semantic labels"""
        return np.array(self._meta["palette"], dtype=np.uint8)

    @property
    def img_sizes(self):
        """Size of each image of the dataset"""
        return [img_desc["size"] for img_desc in self._images]

    @property
    def img_categories(self):
        """Categories present in each image of the dataset"""
        return [img_desc["cat"] for img_desc in self._images]

    @property
    def get_images(self):
        """Categories present in each image of the dataset"""
        return self._images
    @staticmethod
    def numpy_to_torch(s):
        """
        numpy转tensor
        """
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))
    def __len__(self):
        return len(self._images)

    def __getitem__(self, item):
        img_name = self._images[item]
        img_path, label_path, ins_mask_path = self._get_path(img_name)
        image = Image(img_path)
        label = GraspMat(label_path,ins_mask_path,self.mode)

        # rotate
        rota = 30
        rota = np.random.uniform(-1 * rota, rota)
        image.rotate(rota)
        label.rotate(rota)
        # crop
        dist = 5  # 50
        crop_bbox = image.crop(self.crop_size, dist)
        label.crop(crop_bbox) #500,500
        # flip
        flip = True if np.random.rand() < 0.5 else False
        if flip:
            image.flip()
            label.flip()

        # color
        image.color()
        # img归一化
        image.nomalise()
        img = cv2.cvtColor(image.img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1))
        # 获取target
        label.decode(angle_cls=self.angle_k)  # 4通道标签变122
        target = label.grasp  # (2 + angle_k, 500, 500)

        img = self.numpy_to_torch(img)  # 3,500,500 under1
        target = self.numpy_to_torch(target)  # 122,500,500 tensor 0 or 1

        # resize raters images for generating initial point click
        newsize = (self.img_size, self.img_size)
        click_mask_np = cv2.resize(target[0].numpy().copy(), newsize, interpolation=cv2.INTER_NEAREST)

        if 'click' in self.prompt and click_mask_np.max() != 0:
            point_label = 1
            ##########
            box_mask = torch.from_numpy(label.box_mask).unsqueeze(0)
            box_mask = F.interpolate(box_mask, size=(self.img_size, self.img_size), mode="nearest")
            box_mask = torch.as_tensor(box_mask > 0.0, dtype=torch.float32)
            click_mask = box_mask.squeeze().numpy()
            ########
            point_label, pt = random_click(click_mask, point_label)  # in: inputsize,input 0-1
            x = pt[1]
            y = pt[0]
            pt[0] = x
            pt[1] = y
            # point_label, pt_disc = random_click(np.array(target[0]) , point_label)
        else:
            # you may want to get rid of click prompts
            point_label = -1
            pt = np.array([0, 0], dtype=np.int32)
            # print(img_path)
            # print(label.err)

        # first click is the target agreement among most raters

        if self.transform:
            state = torch.get_rng_state()
            # ---------------nomalize----------#
            # 假设 img 是一个形状为 [3, 320, 320]，值范围为 -0.6 到 0.18 的张量
            min_val = img.min()
            max_val = img.max()
            # 将图像数据归一化到 [0, 1]
            img = (img - min_val) / (max_val - min_val)
            # ---------------------------------#
            img = self.transform(TF.to_pil_image(img))  # 3,320,320 tensor to input size  0-1  ------------#-----------

            # transform to mask size (out_size) for mask define
            # conf_mask = (self.transform(target[:1]) > 0.5).float()
            conf_mask = F.interpolate(target[:1].unsqueeze(0), size=(self.mask_size, self.mask_size), mode='nearest',
                                      )  #

            angle_mask = F.interpolate(target[1:-1].unsqueeze(0), size=(self.mask_size, self.mask_size), mode='nearest',
                                       )  #

            width_mask = F.interpolate(target[-1:].unsqueeze(0), size=(self.mask_size, self.mask_size), mode='nearest',
                                       )  # 1,1,512,512   for angle and width , not sure to use bilinear or nearest
            torch.set_rng_state(state)

            mask = torch.concat([conf_mask.squeeze(0)
                                    , angle_mask.squeeze(0)
                                    , width_mask.squeeze(0)
                                 ], dim=0)  # 122,512,512

        if 'box' in self.prompt and conf_mask.max() != 0:
            box_mask = torch.from_numpy(label.box_mask).unsqueeze(0)
            box_mask = F.interpolate(box_mask, size=(self.img_size, self.img_size), mode="nearest")
            box_mask = torch.as_tensor(box_mask > 0.0, dtype=torch.float32)
            x_min_cup, x_max_cup, y_min_cup, y_max_cup = random_box(box_mask)  # ymin,ymax,xmin,xmax
            xmin = y_min_cup
            ymin = x_min_cup
            xmax = y_max_cup
            ymax = x_max_cup
            box_cup = [xmin, ymin, xmax, ymax]


        else:
            # you may want to get rid of box prompts
            box_cup = [0, 0, 0, 0]
            # print('conf_mask.max() = 0',img_path,label.err)


        # ------visualize-------#
        # display_image_with_point(img.permute(1,2,0),pt,box_cup)
        # conf_angle = mask[:2]
        # angle_width = mask[-2:]
        # combined_slices = torch.cat((conf_angle, angle_width), dim=0)
        # label_resized = F.interpolate(combined_slices.unsqueeze(1),size=(self.img_size,self.img_size),mode="nearest")
        # display_labels_with_point(label_resized.squeeze(1),pt,box_cup)
        # ----------------------#

        image_meta_dict = {'filename_or_obj': img_path.split('/')[-1]}
        return {
            'image': img,
            'label': mask,  # 0-1
            'p_label': point_label,
            'pt': pt,
            'box': box_cup,
            'image_meta_dict': image_meta_dict,
        }


    def get_raw_image(self, idx):
        """Load a single, unmodified image with given id frcom the dataset"""
        img_file = os.path.join(self._img_dir, idx)
        if os.path.exists(img_file + ".png"):
            img_file = img_file + ".png"
        elif os.path.exists(img_file + ".jpg"):
            img_file = img_file + ".jpg"
        else:
            raise IOError("Cannot find any image for id {} in {}".format(idx, self._img_dir))

        return Image.open(img_file)

    def get_image_desc(self, idx):
        """Look up an image descriptor given the id"""
        matching = [img_desc for img_desc in self._images if img_desc["id"] == idx]
        if len(matching) == 1:
            return matching[0]
        else:
            raise ValueError("No image found with id %s" % idx)






if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('-prompt', nargs='+', type=str, default=['click','box'], help='Enter one or more valid options.')
        parser.add_argument('-image_size', type=str, default=512, help='input size')
        parser.add_argument('-crop_size', type=str, default=500, help='out size')
        parser.add_argument('-out_size', type=str, default=256, help='out size')

        opt = parser.parse_args()
        return opt
    args = parse_args()
    angle_cls = 120
    data_path = './GraspDetSeg_CNN'
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
    OCID_train_dataset = OCID(args,transform_train,split_name = 'data_split/training_0', data_path = '/data1/samgrasp/dataset/OCID/OCID_grasp')
    train_data = torch.utils.data.DataLoader(
        OCID_train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1)

    print('>> dataset: {}'.format(len(train_data)))

    count = 0
    max_w = 0
    for det in train_data:
        pass