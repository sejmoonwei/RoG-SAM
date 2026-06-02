#!/usr/bin/env	python3

import os
import time

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

import cfg
import function
from conf import settings
from dataset import get_dataloader
from utils import create_logger, get_network, save_checkpoint, set_log_dir
args = cfg.parse_args()

SUPPORTED_DATASETS = {'Cornell', 'OCID'}
if args.dataset not in SUPPORTED_DATASETS:
    raise ValueError(f"Available training datasets: {sorted(SUPPORTED_DATASETS)}.")

GPUdevice = torch.device('cuda', args.gpu_device)

net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)


if args.pretrain:
    checkpoint = torch.load(args.pretrain, map_location=GPUdevice)
    weights = checkpoint.get('state_dict', checkpoint)
    net.load_state_dict(weights, strict=False)

optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

if args.weights != 0:
    print(f'=> resuming from {args.weights}')
    assert os.path.exists(args.weights)
    checkpoint_file = os.path.join(args.weights)
    assert os.path.exists(checkpoint_file)
    loc = 'cuda:{}'.format(args.gpu_device)
    checkpoint = torch.load(checkpoint_file, map_location=loc)
    start_epoch = checkpoint['epoch']
    best_tol = checkpoint['best_tol']
    
    net.load_state_dict(checkpoint['state_dict'],strict=False)
    args.path_helper = checkpoint['path_helper']
    logger = create_logger(args.path_helper['log_path'])
    print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')

args.path_helper = set_log_dir('logs', args.exp_name)
logger = create_logger(args.path_helper['log_path'])
logger.info(args)

nice_train_loader, nice_test_loader = get_dataloader(args)

checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
if not os.path.exists(settings.LOG_DIR):
    os.mkdir(settings.LOG_DIR)
writer = SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, args.net, settings.TIME_NOW))
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

for epoch in range(settings.EPOCH):
    if epoch and epoch < 5:
        accuracy = function.validation_sam(args, nice_test_loader, epoch, net, writer)
        logger.info(f'Accuracy: {accuracy}|| @ epoch {epoch}.')

    net.train()
    time_start = time.time()
    loss = function.train_sam(args, net, optimizer, nice_train_loader, epoch, writer, vis=args.vis)
    logger.info(f'Train loss: {loss} || @ epoch {epoch}.')
    time_end = time.time()
    print('time_for_training ', time_end - time_start)

    net.eval()
    if epoch and epoch % args.val_freq == 0 or epoch == settings.EPOCH-1:
        accuracy = function.validation_sam(args, nice_test_loader, epoch, net, writer)
        logger.info(f'Accuracy: {accuracy}|| @ epoch {epoch}.')

        if args.distributed != 'none':
            sd = net.module.state_dict()
        else:
            sd = net.state_dict()

        if True:
            is_best = True

            save_checkpoint({
            'epoch': epoch + 1,
            'model': args.net,
            'state_dict': sd,
            'optimizer': optimizer.state_dict(),
            'path_helper': args.path_helper,
        }, is_best, args.path_helper['ckpt_path'], filename="best_dice_checkpoint.pth",epoch = epoch)
            print("model saved")
        else:
            is_best = False

writer.close()
