import argparse
import os
import sys
import time
from collections import OrderedDict
import torchvision.transforms.functional as TF
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from skimage import io
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from tensorboardX import SummaryWriter
#from dataset import *
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
import cfg
import function
from conf import settings
#from models.discriminatorlayer import discriminator
from dataset import *
from utils import *
import matplotlib.pyplot as plt
from util.data.structure.img import Image
from util.data.structure.grasp import GraspMat, drawGrasp1
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import cv2
import matplotlib.patches as patches




def display_image_with_point(image, point= [0,0], box = [0,0,0,0]):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.scatter(point[0], point[1], color='red', s=100)  # 点的大小为100

    # 添加边界框
    rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    plt.axis("off")
    plt.show()


def display_labels_with_point(data, point= [0,0], box = [0,0,0,0]):
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    titles = ['Label 1'
        # , 'Label 2'
        # , 'Label 3'
        # , 'Label 4'
              ]


    axs.imshow(data[0], cmap='gray')
    axs.scatter(point[0], point[1], color='red', s=100)  # 在每个标签上标记点

    # 在每个子图上添加边界框
    rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
    axs.add_patch(rect)

    axs.set_title(titles[0])
    axs.axis('off')

    plt.tight_layout()
    plt.show()

def numpy_to_torch(s):
    """
    numpy转tensor
    """
    if len(s.shape) == 2:
        return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
    else:
        return torch.from_numpy(s.astype(np.float32))

transform_train = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
])

pretrain = '/data1/ori/checkpoint/Jacquard5.pth'
args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
image_size = 512
mask_size = 256
net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
print('Network build')
weights = torch.load(pretrain, map_location='cuda:0')['state_dict']
net.load_state_dict(weights,strict=True)
# for name, param in weights.items():
#     if name in weights:
#         net.state_dict()[name].copy_(param)


random_tensor = torch.rand(1,3,image_size,image_size)
file = '/data1/samgrasp/dataset/cornell_adapt/Training/pcd0388r.png'
label_path = file.replace('r.png', 'grasp.mat')



# raw image and raters images
image = Image(file)
label = GraspMat(label_path)

#-----argument------#
# crop
dist = 10  # 50
crop_bbox = image.crop(500, dist)
label.crop(crop_bbox)
label.decode(angle_cls=120)  # 4通道标签变122


# color
image.color()
#------------------#
# img归一化
image.nomalise()

img = cv2.cvtColor(image.img,cv2.COLOR_BGR2RGB)
img = img.transpose((2, 0, 1))  # (320, 320, 3) -> (3, 320, 320)
img = numpy_to_torch(img)  # 3,320,320 under1
# ---------------nomalize----------

# 假设 img 是一个形状为 [3, 320, 320]，值范围为 -0.6 到 0.18 的张量
min_val = img.min()
max_val = img.max()

# 将图像数据归一化到 [0, 1]
img = (img - min_val) / (max_val - min_val)
# ---------------------------------#
img = transform_train(TF.to_pil_image(img))

# display_image(img.permute(1,2,0))

img_in = img.unsqueeze(0).to(GPUdevice)

#-------------get prompt--------------#
target = label.grasp  # (2 + angle_k, 320, 320)
target = numpy_to_torch(target)  # 122,320,320 tensor 0 or 1

conf_mask = F.interpolate(target[:1].unsqueeze(0), size=(mask_size, mask_size), mode='bilinear',
                          align_corners=False)  #

angle_mask = F.interpolate(target[1:-1].unsqueeze(0), size=(mask_size, mask_size), mode='nearest',
                           align_corners=False)  #

width_mask = F.interpolate(target[-1:].unsqueeze(0), size=(mask_size, mask_size), mode='nearest',
                           align_corners=False)  # 1,1,512,512

mask = torch.concat([conf_mask.squeeze(0)
                     # ,angle_mask.squeeze(0)
                     # , width_mask.squeeze(0)
                     ], dim=0)  # 122,512,512

newsize = (mask_size, mask_size)
click_mask_np = cv2.resize(target[0].numpy().copy(), (image_size,image_size), interpolation=cv2.INTER_LINEAR)
point_label, pt = random_click(click_mask_np, point_labels = 1)
x = pt[1]
y = pt[0]
pt[0] = x
pt[1] = y
pt_coord = pt.copy()
point_label = torch.tensor([point_label])
pt = torch.tensor([pt])
if point_label.clone().flatten()[0] != -1:
    # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
    point_coords = pt  # 390 339
    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
    labels_torch = torch.as_tensor(point_label, dtype=torch.int, device=GPUdevice)
    if (len(point_label.shape) == 1):  # only one point prompt
        # coords_torch, labels_torch, showp = coords_torch[None, :, :], labels_torch[None, :], showp[None, :, :]
        coords_torch, labels_torch = coords_torch[:, None, :], labels_torch[:, None]
    pt = (coords_torch, labels_torch)

box_mask = F.interpolate(conf_mask, size=(image_size, image_size), mode="bilinear")
box_mask = torch.as_tensor(box_mask > 0.5, dtype=torch.float32)
x_min_cup, x_max_cup, y_min_cup, y_max_cup = random_box(box_mask)  # ymin,ymax,xmin,xmax
xmin = y_min_cup
ymin = x_min_cup
xmax = y_max_cup
ymax = x_max_cup
# box_cup = [x_min_cup, x_max_cup, y_min_cup, y_max_cup]
box_cup = [xmin, ymin, xmax, ymax]
box = [torch.tensor(box_cup)]
combined_box = torch.stack(box, dim=0).to(dtype=torch.float32, device=GPUdevice)
combined_box = combined_box[:, None, :]






with torch.no_grad():
    imge = net.image_encoder(img_in)
    if args.net == 'sam' or args.net == 'mobile_sam':
        se, de = net.prompt_encoder(
            points= pt, #pt,
            boxes= combined_box, #combined_box,
            masks=None,
        )
    if args.net == 'sam':
        pred, _ = net.mask_decoder(
            image_embeddings=imge,
            image_pe=net.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=se,
            dense_prompt_embeddings=de,
            multimask_output=(args.multimask_output > 1),
        )
    pred = pred.squeeze(0)
    conf_angle = pred[:1]
    angle = pred[1:-1]
    angle_width = pred[-1:]
    able_pred = torch.sigmoid(conf_angle)
    angle_pred = torch.sigmoid(angle)
    width_pred = torch.sigmoid(angle_width)
    combined_slices = torch.cat((able_pred
                                 # , angle_pred
                                 , angle_width
                                 ), dim=0)
    # combined_slices = torch.cat((able_pred, width_pred), dim=0)

    # ------visualize-------#
    # img_to_show = (image.img - image.img.min()) / (image.img.max() - image.img.min())
    display_image_with_point(img_in.squeeze(0).permute(1,2,0).cpu().numpy(),pt_coord,box_cup)  # 320,320,3   255
    lbl = torch.from_numpy(label.grasp[:1]).unsqueeze(0)
    label_to_vis = F.interpolate(lbl, size=(image_size, image_size), mode="bilinear", align_corners=False)
    display_labels_with_point(label_to_vis.squeeze(0),pt_coord,box_cup)  # 4,320,320
    # ----------------------#
    # display_labels(combined_slices.cpu())
    pred_ = torch.as_tensor(combined_slices > 0.5, dtype=torch.float32).cpu()
    display_labels_with_point(pred_[:1])
    pass


























