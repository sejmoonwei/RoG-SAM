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
import torch.utils.data
import torch.optim as optim
from util.data.evaluation_graspnet import evaluation
from util.data import get_dataset
from util.saver import Saver
from dataset import Graspnet
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

def validate(net, device, val_data, args, top_k):
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
        'None': 0,
        'loss': 0,
        'graspable': 0,
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

            ret = evaluation(able_out, angle_out, width_out, y, top_k)  # y:1,122,320,320 tensor
            if ret == True:
                results['correct'] += 1
                # tqdm.write('correct: {}'.format(results['correct']))
            elif ret == False:
                results['failed'] += 1
                # tqdm.write('failed: {}'.format(results['failed']))
                results['fail'].append(batch_idx)
            elif ret == None:
                results['None'] += 1

    return results

def main():
    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    GPUdevice = torch.device('cuda', args.gpu_device)

    '''graspnet data'''

    Graspnet_test_seen_dataset = Graspnet(args, transform_test, camera=args.camera,
                                          data_path='/data/myp/grasp_dataset', split='test_seen')
    Graspnet_test_similar_dataset = Graspnet(args, transform_test, camera=args.camera,
                                             data_path='/data/myp/grasp_dataset', split='test_similar')
    Graspnet_test_novel_dataset = Graspnet(args, transform_test, camera=args.camera,
                                           data_path='/data/myp/grasp_dataset', split='test_novel')


    nice_test_seen_loader = DataLoader(Graspnet_test_seen_dataset, batch_size=1, shuffle=False, num_workers=8,
                                       pin_memory=True)
    nice_test_similar_loader = DataLoader(Graspnet_test_similar_dataset, batch_size=1, shuffle=False, num_workers=8,
                                          pin_memory=True)
    nice_test_novel_loader = DataLoader(Graspnet_test_novel_dataset, batch_size=1, shuffle=False, num_workers=8,
                                        pin_memory=True)
    '''end'''



    pretrain = './checkpoint/kinect_11_08_11_04_31.pth'

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
    weights = torch.load(pretrain, map_location=GPUdevice)['state_dict']
    # net.load_state_dict(weights,strict=True)
    for name, param in weights.items():
        if name in weights:
            net.state_dict()[name].copy_(param)

    import csv
    import sys

    # 创建日志文件，重定向 print 输出
    log_file = open('evaluation_log_miu1.txt', 'w')
    sys.stdout = log_file

    # 打开 CSV 文件并写入表头
    with open('resultsmiu1.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['top_k', 'split', 'graspable', 'correct', 'total', 'accuracy'])

        for top_k in range(1, 51, 2):
            # ========== Seen Split ==========
            train_val_results = validate(net, GPUdevice, nice_test_seen_loader, args, top_k)
            graspable_seen = train_val_results['graspable']
            correct_seen = train_val_results['correct']
            total_seen = train_val_results['correct'] + train_val_results['failed']
            acc_seen = correct_seen / total_seen

            print(
                f'Top-{top_k} | Seen Split | Graspable: {graspable_seen:.5f} | Accuracy: {correct_seen}/{total_seen} = {acc_seen:.5f}')
            writer.writerow([top_k, 'seen', graspable_seen, correct_seen, total_seen, acc_seen])

            # ========== Similar Split ==========
            train_val_results = validate(net, GPUdevice, nice_test_similar_loader, args, top_k)
            graspable_similar = train_val_results['graspable']
            correct_similar = train_val_results['correct']
            total_similar = train_val_results['correct'] + train_val_results['failed']
            acc_similar = correct_similar / total_similar

            print(
                f'Top-{top_k} | Similar Split | Graspable: {graspable_similar:.5f} | Accuracy: {correct_similar}/{total_similar} = {acc_similar:.5f}')
            writer.writerow([top_k, 'similar', graspable_similar, correct_similar, total_similar, acc_similar])

            # ========== Novel Split ==========
            train_val_results = validate(net, GPUdevice, nice_test_novel_loader, args, top_k)
            graspable_novel = train_val_results['graspable']
            correct_novel = train_val_results['correct']
            total_novel = train_val_results['correct'] + train_val_results['failed']
            acc_novel = correct_novel / total_novel

            print(
                f'Top-{top_k} | Novel Split | Graspable: {graspable_novel:.5f} | Accuracy: {correct_novel}/{total_novel} = {acc_novel:.5f}')
            writer.writerow([top_k, 'novel', graspable_novel, correct_novel, total_novel, acc_novel])

    # 恢复标准输出
    sys.stdout = sys.__stdout__

    # 关闭日志文件
    log_file.close()


if __name__ == "__main__":
    main()



























