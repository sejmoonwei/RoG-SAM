import datetime
import os
import sys
import argparse
import logging
import cv2
import time
import numpy as np
import random
import torch
import torch.utils.dataf
import torch.optim as optim
from util.data.evaluation_OCID import evaluation
from util.data import get_dataset
from util.saver import Saver
from dataset import OCID
from torch.utils.data import DataLoader, random_split
from cfg import parse_args
logging.basicConfig(level=logging.INFO)
import torchvision.transforms as transforms
from utils import get_network
import torch.nn.functional as F
from skimage.filters import gaussian
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

args = parse_args()


def vis_img(img_np, box, pt):
    xmin, ymin, xmax, ymax = box[0].item(), box[1].item(), box[2].item(), box[3].item()
    x_coord, y_coord = pt[0, 0], pt[0, 1]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_np)
    ax.axis('off')
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r',
                             facecolor='none')
    ax.add_patch(rect)
    ax.plot(x_coord, y_coord, 'bo')
    plt.show()


def vis_mask(conf_np, width_np):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(conf_np, cmap='gray')
    axes[0].set_title('conf')
    axes[0].axis('off')
    axes[1].imshow(width_np, cmap='gray')
    axes[1].set_title('width')
    axes[1].axis('off')
    plt.show()


def post_process_output(able_pred, angle_pred, width_pred):
    """
    :param able_pred:  (1, 2, 320, 320)      (as torch Tensors)
    :param angle_pred: (1, angle_k, 320, 320)     (as torch Tensors)
    """

    # 抓取置信度
    able_pred = able_pred.squeeze().cpu().numpy()    # (320, 320)
    able_pred = gaussian(able_pred, 1.0, preserve_range=True)

    # 抓取角
    angle_pred = np.argmax(angle_pred.cpu().numpy().squeeze(), 0)   # (320, 320)

    # 抓取宽度
    width_pred = width_pred.squeeze().cpu().numpy() * 100.  # (320, 320)
    width_pred = gaussian(width_pred, 1.0, preserve_range=True)

    return able_pred, angle_pred, width_pred

def compute_loss(net, xc, target, pt, combined_box, device):
    able_target = target[:, 0, :, :].to(device)        # (-1, 256, 256)
    angle_target = target[:, 1:-1, :, :].to(device)    # (-1, angle_k, 256, 256)
    width_target = target[:, -1, :, :].to(device)      # (-1, 256, 256)


    imge = net.image_encoder(xc)
    if args.net == 'sam' or args.net == 'mobile_sam':
        try:
            se, de = net.prompt_encoder(
                points= pt,
                boxes=combined_box,
                masks=None,
            )
        except ValueError:
            print('Error pt:',pt)



    if args.net == 'sam':
        pred, _ = net.mask_decoder(
            image_embeddings=imge,
            image_pe=net.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=se,
            dense_prompt_embeddings=de,
            multimask_output=(args.multimask_output > 1),
        )
    pred = F.interpolate(pred, size=(args.out_size, args.out_size))

    pred = pred.squeeze(0)
    conf = pred[:1] #1,128,128
    angle = pred[1:-1] #120,128,128
    width = pred[-1:] #1,128,128


    # 置信度损失
    able_pred = torch.sigmoid(conf)
    able_loss = F.binary_cross_entropy(able_pred.squeeze(), able_target.squeeze()) #able_pred.squeeze() 2,320,320  0.48

    # 抓取角损失
    angle_pred = torch.sigmoid(angle)
    angle_loss = F.binary_cross_entropy(angle_pred.squeeze(), angle_target.squeeze()) #angle_pred.squeeze() 2,120,320,320 0.72

    # 抓取宽度损失
    width_pred = torch.sigmoid(width)
    width_loss = F.binary_cross_entropy(width_pred.squeeze(), width_target.squeeze())

    return {
        'loss': able_loss + angle_loss * 10 + width_loss,
        'losses': {
            'able_loss': able_loss,
            'angle_loss': angle_loss * 10,
            'width_loss': width_loss,
        },
        'pred': {
            'able': able_pred,      # [-1, 1, 320, 320]
            'angle': angle_pred,    # [-1, angle_k, 320, 320]
            'width': width_pred,    # [-1, 1, 320, 320]
        }
    }


def min_max_normalization(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr

def validate(net, device, val_data, args):
    """
    Run validation.
    :param net: 网络
    :param device:
    :param val_data: 验证数据集
    :param saver: 保存器
    :param args:
    :return: Successes, Failures and Losses
    """
    net.eval()

    results = {
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'graspable': 0,
        'Non': 0,
        'fail': [],
        'losses': {
        }
    }

    ld = len(val_data)

    with torch.no_grad():  # 不计算梯度，不反向传播
        for batch_idx, pack in enumerate(tqdm(val_data, desc="Validating", unit="batch")):
            x = pack['image']  # 1,3,512,512 tensor
            y = pack['label']  # 1,122,256,256
            pt = pack['pt']
            point_labels = pack['p_label']
            showp = pt

            if True:
                    # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                point_coords = pt #390 339
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=device)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=device)
                if(len(point_labels.shape)==1): # only one point prompt
                    # coords_torch, labels_torch, showp = coords_torch[None, :, :], labels_torch[None, :], showp[None, :, :]
                    coords_torch, labels_torch, showp = coords_torch[:, None, :], labels_torch[:, None], showp[:, None, :]

                pt = (coords_torch, labels_torch)

            box = pack['box']  # list 4 each tensor 4
            combined_box = torch.stack(box, dim=0).to(dtype=torch.float32, device=device)
            combined_box = combined_box[:, None, :]  # 4,1,4

            lossd = compute_loss(net, x.to(device), y.to(device), pt, combined_box, device) #4,3,512,512 4,122,256,256

            # 统计损失
            loss = lossd['loss']  # 损失和
            results['loss'] += loss.item() / ld  # 损失累加
            for ln, l in lossd['losses'].items():  # 添加单项损失
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item() / ld

            # 输出值预处理
            able_out, angle_out, width_out = post_process_output(
                lossd['pred']['able'], lossd['pred']['angle'], lossd['pred']['width']
            )  # in: 1,1,320,320 tensor out: 320,320 array

            # 评估
            results['graspable'] += np.max(able_out) / ld

            ret = evaluation(able_out, angle_out, width_out, y)  # y:1,122,320,320 tensor
            if ret == True :
                results['correct'] += 1
                # tqdm.write('correct: {}'.format(results['correct']))
            elif ret == False :
                results['failed'] += 1
                # tqdm.write('failed: {}'.format(results['failed']))
                results['fail'].append(batch_idx)
            elif ret == None :
                results['Non'] += 1

    return results
# def validate(net, device, val_data, args):
#     """
#     Run validation.
#     :param net: 网络
#     :param device:
#     :param val_data: 验证数据集
#     :param saver: 保存器
#     :param args:
#     :return: Successes, Failures and Losses
#     """
#     net.eval()
#
#     results = {
#         'correct': 0,
#         'failed': 0,
#         'none': 0,
#         'loss': 0,
#         'graspable': 0,
#         'fail': [],
#         'losses': {
#         }
#     }
#
#     ld = len(val_data)
#
#     with torch.no_grad():     # 不计算梯度，不反向传播
#         batch_idx = 0
#         for pack in val_data:
#             x = pack['image'] #1,3,320,320 tensor
#             y = pack['label'] #1,122,320,320
#             pt = pack['pt']
#             point_labels = pack['p_label']
#             box = pack['box']
#             showp = pt
#
#             # if point_labels.clone().flatten()[0] != -1:
#             if True:
#                 # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
#                 point_coords = pt  # 390 339
#                 coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=device)
#                 labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=device)
#                 if (len(point_labels.shape) == 1):  # only one point prompt
#                     coords_torch, labels_torch, showp = coords_torch[None, :, :], labels_torch[None, :], showp[None, :, :]
#                 pt = (coords_torch, labels_torch)
#
#             batch_idx += 1
#             print ("\r Validating... {:.2f}".format(batch_idx/ld), end="")
#
#
#             lossd = compute_loss(net, x.to(device), y.to(device), pt, device)
#
#
#
#             # 统计损失
#             loss = lossd['loss']    # 损失和
#             results['loss'] += loss.item()/ld       # 损失累加
#             for ln, l in lossd['losses'].items():   # 添加单项损失
#                 if ln not in results['losses']:
#                     results['losses'][ln] = 0
#                 results['losses'][ln] += l.item()/ld
#
#             # 输出值预处理
#             able_out, angle_out, width_out = post_process_output(lossd['pred']['able'], lossd['pred']['angle'], lossd['pred']['width']) #in: 1,1,320,320 tensor out : 320,320 array
#             #---------------vis processed pred here------------------#
#
#             #-------------------------------------------------------#
#
#             # able_out = min_max_normalization(able_out)   #fixed  important
#             # 保存预测图
#             # saver.save_img(epoch, batch_idx, [able_out_0, able_out_1, yc])
#
#             # 评估
#             results['graspable'] += np.max(able_out)/ld
#
#
#
#             ret = evaluation(able_out, angle_out, width_out, y) #y:1,122,320,320 tensor
#             #---------------------vis fail------------------#
#             # if ret == 'False':
#             if True:
#                 pass
#                 #------------vis pred mask here-----------------#
#                 # img_np = x.numpy()
#                 # img_np = np.squeeze(img_np, axis=0)
#                 # img_np = np.transpose(img_np, (1, 2, 0))
#                 # vis_img(img_np, pack['box'], pack['pt'])
#                 # label_np = y.numpy()
#                 # label_np = np.squeeze(label_np, axis=0)
#                 # mask_conf = label_np[0]
#                 # mask_width = label_np[121]
#                 # vis_mask(mask_conf, mask_width)
#                 # # -----------------pred----------------------#
#                 # conf = lossd['pred']['able'].cpu()
#                 # width = lossd['pred']['width'].cpu()
#                 # conf_pred_np = conf.numpy().transpose(1, 2, 0)
#                 # width_pred_np = width.numpy().transpose(1, 2, 0)
#                 # vis_mask(conf_pred_np, width_pred_np)
#                 #------------------------------------------------#
#
#
#             if ret == 'Correct':
#                 results['correct'] += 1
#                 print('correct:',results['correct'])
#             elif ret == 'False':
#                 results['failed'] += 1
#                 print('failed:',results['failed'])
#                 results['fail'].append(batch_idx)
#             elif ret == 'None':
#                 results['none'] += 1
#                 print('None:',results['none'])
#
#     return results
def main():

    transform_train = transforms.Compose([
        transforms.Resize((args.image_size,args.image_size)),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    GPUdevice = torch.device('cuda', args.gpu_device)

    '''OCID data'''
    OCID_test_dataset = OCID(args, transform=transform_test, split_name='data_split/validation_0', data_path=args.data_path)
    nice_test_loader = DataLoader(OCID_test_dataset, batch_size=1, shuffle=False, num_workers=8,
                                  pin_memory=True)
    '''end'''



    pretrain = '/data1/ori/checkpoint/OCIDsam902.pth'

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
    weights = torch.load(pretrain, map_location=GPUdevice)['state_dict']
    # net.load_state_dict(weights,strict=True)
    for name, param in weights.items():
        if name in weights:
            net.state_dict()[name].copy_(param)


    train_val_results = validate(net, GPUdevice, nice_test_loader, args)

    print('>>> train_graspable = {:.5f}'.format(train_val_results['graspable']))
    print('>>> train_acc: %d/%d = %f' % (
    train_val_results['correct'], train_val_results['correct'] + train_val_results['failed'],
    train_val_results['correct'] / (train_val_results['correct'] + train_val_results['failed'])))

if __name__ == "__main__":
    main()



























