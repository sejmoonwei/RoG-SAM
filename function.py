
import argparse
import os
import shutil
import sys
import tempfile
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from einops import rearrange
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.transforms import AsDiscrete
from PIL import Image
from skimage import io
from skimage.filters import gaussian
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import cfg
import models.sam.utils.transforms as samtrans
import pytorch_ssim
from conf import settings
from utils import *

args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
seed = torch.randint(1,11,(args.b,7))

torch.backends.cudnn.benchmark = True
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
scaler = torch.cuda.amp.GradScaler()
max_iterations = settings.EPOCH
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

def train_sam(args, net: nn.Module, optimizer, train_loader,
          epoch, writer, schedulers=None, vis = 50):
    hard = 0
    epoch_loss = 0
    ind = 0
    net.train()
    optimizer.zero_grad()

    epoch_loss = 0
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masks = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            if 'pt' not in pack:
                imgs, pt, masks = generate_click_prompt(imgs, masks)
            else:
                pt = pack['pt']
                point_labels = pack['p_label']
            name = pack['image_meta_dict']['filename_or_obj']

            if 'box' in pack and 'box' in args.prompt:
                box = pack['box']
                combined_box = torch.stack(box,dim=0).to(dtype = torch.float32, device = GPUdevice)
                combined_box = combined_box[:,None,:]
            else:
                combined_box = None



            if args.thd:
                imgs, pt, masks = generate_click_prompt(imgs, masks)

                pt = rearrange(pt, 'b n d -> (b d) n')
                imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                masks = rearrange(masks, 'b c h w d -> (b d) c h w ')

                imgs = imgs.repeat(1,3,1,1)
                point_labels = torch.ones(imgs.size(0))

                imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
            showp = pt

            mask_type = torch.float32
            ind += 1
            b_size,c,w,h = imgs.size()
            longsize = w if w >=h else h

            point_coords = pt
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
            if len(point_labels.shape) == 1:
                coords_torch, labels_torch, showp = coords_torch[:, None, :], labels_torch[:, None], showp[:, None, :]

            pt = (coords_torch, labels_torch)

            if hard:
                true_mask_ave = (true_mask_ave > 0.5).float()

            if args.mod == 'sam_adpt':
                for n, value in net.image_encoder.named_parameters(): 
                    if "Adapter" not in n:
                        value.requires_grad = True
                    else:
                        value.requires_grad = False
            elif args.mod == 'sam_lora' or args.mod == 'sam_adalora':
                from models.common import loralib as lora
                lora.mark_only_lora_as_trainable(net.image_encoder)
                if args.mod == 'sam_adalora':
                    rankallocator = lora.RankAllocator(
                        net.image_encoder, lora_r=4, target_rank=8,
                        init_warmup=500, final_warmup=1500, mask_interval=10, 
                        total_step=3000, beta1=0.85, beta2=0.85, 
                    )
            elif args.mod == 'sam':
                for n, value in net.image_encoder.named_parameters(): 
                    value.requires_grad = True


            imge= net.image_encoder(imgs)
            with torch.no_grad():
                if args.net == 'sam' or args.net == 'mobile_sam':
                    se, de = net.prompt_encoder(
                        points=pt,
                        boxes=combined_box,
                        masks=None,
                    )

                elif args.net == "efficient_sam":
                    coords_torch,labels_torch = transform_prompt(coords_torch,labels_torch,h,w)
                    se = net.prompt_encoder(
                        coords=coords_torch,
                        labels=labels_torch,
                    )
                    
            if args.net == 'sam':
                pred, _ = net.mask_decoder(
                    image_embeddings=imge,
                    image_pe=net.prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=(args.multimask_output > 1),
                )
            elif args.net == 'mobile_sam':
                pred, _ = net.mask_decoder(
                    image_embeddings=imge,
                    image_pe=net.prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=False,
                )
            elif args.net == "efficient_sam":
                se = se.view(
                    se.shape[0],
                    1,
                    se.shape[1],
                    se.shape[2],
                )
                pred, _ = net.mask_decoder(
                    image_embeddings=imge,
                    image_pe=net.prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=se,
                    multimask_output=False,
                )
                
            pred = F.interpolate(pred,size=(args.out_size,args.out_size))
            if args.dataset in ('Cornell', 'OCID'):
                pred_able = pred[:,:1,:,:]
                pred_angle = pred[:,1:-1,:,:]
                pred_width =pred[:,-1:,:,:]

                able_target = masks[:,:1,:,:]
                angle_target = masks[:,1:-1,:,:]
                width_target = masks[:,-1:,:,:]

                if args.dataset == 'Cornell':
                    def calculate_pos_weight(target):
                        pos = target.sum()
                        total = target.numel()
                        ratio = (total - pos) / max(pos, 1)
                        return min(ratio,2)
                elif args.dataset == 'OCID':
                    def calculate_pos_weight(target):
                        pos = target.sum()
                        total = target.numel()
                        ratio = (total - pos) / max(pos, 1)
                        return min(ratio, 16)


                pos_weight_able = torch.tensor([calculate_pos_weight(able_target)], dtype=torch.float32).to(GPUdevice)
                pos_weight_angle = torch.tensor([calculate_pos_weight(angle_target)], dtype=torch.float32).to(GPUdevice)
                pos_weight_width = torch.tensor([calculate_pos_weight(width_target)], dtype=torch.float32).to(GPUdevice)

                lossf_able = nn.BCEWithLogitsLoss(pos_weight=pos_weight_able)
                lossf_angle = nn.BCEWithLogitsLoss(pos_weight=pos_weight_angle)
                lossf_width = nn.BCEWithLogitsLoss(pos_weight=pos_weight_width)


                able_loss = lossf_able(pred_able.squeeze(), able_target.squeeze())
                angle_loss = lossf_angle(pred_angle.squeeze(), angle_target.squeeze())
                width_loss = lossf_width(pred_width.squeeze(), width_target.squeeze())



                loss = able_loss * 5 + angle_loss + width_loss
                pbar.set_postfix(**{'loss (batch)': loss.item(),
                                    'able_loss(batch)': able_loss.item(),
                                    'angle_loss(batch)': angle_loss.item(),
                                    'width_loss(batch)': width_loss.item()
                                    })

            else:
                loss = lossfunc(pred, masks)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

            epoch_loss += loss.item()

            if args.mod == 'sam_adalora':
                (loss+lora.compute_orth_regu(net, regu_weight=0.1)).backward()
                optimizer.step()
                rankallocator.update_and_mask(net, ind)
            else:
                loss.backward()
                optimizer.step()
            
            optimizer.zero_grad()

            '''vis images'''
            if vis:
                if ind % vis == 0:
                    namecat = 'Train'
                    for na in name[:2]:
                        namecat = namecat + na.split('/')[-1].split('.')[0] + '+'

                    vis_image(imgs,pred,masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=showp)

            pbar.update()

    return loss

def _get_grasp_evaluation(dataset_name):
    if dataset_name == 'Cornell':
        from util.data.evaluation import evaluation
    elif dataset_name == 'OCID':
        from util.data.evaluation_OCID import evaluation
    else:
        raise ValueError("Available evaluation datasets: 'Cornell' and 'OCID'.")
    return evaluation


def post_process_output(able_pred, angle_pred, width_pred):
    able_pred = able_pred.squeeze().cpu().numpy()
    able_pred = gaussian(able_pred, 1.0, preserve_range=True)

    angle_pred = np.argmax(angle_pred.cpu().numpy().squeeze(), 0)

    width_pred = width_pred.squeeze().cpu().numpy() * 100.0
    width_pred = gaussian(width_pred, 1.0, preserve_range=True)

    return able_pred, angle_pred, width_pred


def prepare_prompt(pt, point_labels, device):
    point_coords = pt
    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=device)
    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=device)

    if labels_torch.clone().flatten()[0] == -1:
        return None

    if len(labels_torch.shape) == 1:
        coords_torch = coords_torch[:, None, :]
        labels_torch = labels_torch[:, None]

    return coords_torch, labels_torch


def prepare_box(box, device):
    if box is None:
        return None
    combined_box = torch.stack(box, dim=0).to(dtype=torch.float32, device=device)
    return combined_box[:, None, :]


def compute_grasp_loss(net, xc, target, pt, combined_box, device, run_args):
    able_target = target[:, 0, :, :].to(device)
    angle_target = target[:, 1:-1, :, :].to(device)
    width_target = target[:, -1, :, :].to(device)

    imge = net.image_encoder(xc)
    se, de = net.prompt_encoder(
        points=pt,
        boxes=combined_box,
        masks=None,
    )
    pred, _ = net.mask_decoder(
        image_embeddings=imge,
        image_pe=net.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=se,
        dense_prompt_embeddings=de,
        multimask_output=(run_args.multimask_output > 1),
    )
    pred = F.interpolate(pred, size=(run_args.out_size, run_args.out_size))

    pred = pred.squeeze(0)
    conf = pred[:1]
    angle = pred[1:-1]
    width = pred[-1:]

    able_pred = torch.sigmoid(conf)
    angle_pred = torch.sigmoid(angle)
    width_pred = torch.sigmoid(width)

    able_loss = F.binary_cross_entropy(able_pred.squeeze(), able_target.squeeze())
    angle_loss = F.binary_cross_entropy(angle_pred.squeeze(), angle_target.squeeze())
    width_loss = F.binary_cross_entropy(width_pred.squeeze(), width_target.squeeze())

    return {
        'loss': able_loss + angle_loss * 10 + width_loss,
        'losses': {
            'able_loss': able_loss,
            'angle_loss': angle_loss * 10,
            'width_loss': width_loss,
        },
        'pred': {
            'able': able_pred,
            'angle': angle_pred,
            'width': width_pred,
        },
    }


def validate(net, device, val_data, run_args):
    net.eval()
    evaluation = _get_grasp_evaluation(run_args.dataset)
    results = {
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'graspable': 0,
        'Non': 0,
        'fail': [],
        'losses': {},
    }

    ld = len(val_data)
    with torch.no_grad():
        for batch_idx, pack in enumerate(tqdm(val_data, desc='Validating', unit='batch')):
            x = pack['image']
            y = pack['label']
            pt = prepare_prompt(pack['pt'], pack['p_label'], device)
            combined_box = prepare_box(pack.get('box'), device)

            lossd = compute_grasp_loss(net, x.to(device), y.to(device), pt, combined_box, device, run_args)
            loss = lossd['loss']
            results['loss'] += loss.item() / ld
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item() / ld

            able_out, angle_out, width_out = post_process_output(
                lossd['pred']['able'],
                lossd['pred']['angle'],
                lossd['pred']['width'],
            )
            results['graspable'] += np.max(able_out) / ld

            ret = evaluation(able_out, angle_out, width_out, y)
            if ret is True:
                results['correct'] += 1
            elif ret is False:
                results['failed'] += 1
                results['fail'].append(batch_idx)
            elif ret is None:
                results['Non'] += 1

    return results


def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
    device = torch.device('cuda', args.gpu_device)
    train_val_results = validate(net, device, val_loader, args)

    total = train_val_results['correct'] + train_val_results['failed']
    accuracy = train_val_results['correct'] / total if total else 0.0
    print('>>> train_graspable = {:.5f}'.format(train_val_results['graspable']))
    print('>>> train_acc: %d/%d = %f' % (
        train_val_results['correct'],
        total,
        accuracy,
    ))
    if train_val_results.get('Non'):
        print('>>> ignored_none: {}'.format(train_val_results['Non']))

    return accuracy


def transform_prompt(coord,label,h,w):
    coord = coord.transpose(0,1)
    label = label.transpose(0,1)

    coord = coord.unsqueeze(1)
    label = label.unsqueeze(1)

    batch_size, max_num_queries, num_pts, _ = coord.shape
    num_pts = coord.shape[2]
    rescaled_batched_points = get_rescaled_pts(coord, h, w)

    decoder_max_num_input_points = 6
    if num_pts > decoder_max_num_input_points:
        rescaled_batched_points = rescaled_batched_points[
            :, :, : decoder_max_num_input_points, :
        ]
        label = label[
            :, :, : decoder_max_num_input_points
        ]
    elif num_pts < decoder_max_num_input_points:
        rescaled_batched_points = F.pad(
            rescaled_batched_points,
            (0, 0, 0, decoder_max_num_input_points - num_pts),
            value=-1.0,
        )
        label = F.pad(
            label,
            (0, decoder_max_num_input_points - num_pts),
            value=-1.0,
        )
    
    rescaled_batched_points = rescaled_batched_points.reshape(
        batch_size * max_num_queries, decoder_max_num_input_points, 2
    )
    label = label.reshape(
        batch_size * max_num_queries, decoder_max_num_input_points
    )

    return rescaled_batched_points,label


def get_rescaled_pts(batched_points: torch.Tensor, input_h: int, input_w: int):
        return torch.stack(
            [
                torch.where(
                    batched_points[..., 0] >= 0,
                    batched_points[..., 0] * 1024 / input_w,
                    -1.0,
                ),
                torch.where(
                    batched_points[..., 1] >= 0,
                    batched_points[..., 1] * 1024 / input_h,
                    -1.0,
                ),
            ],
            dim=-1,
        )
