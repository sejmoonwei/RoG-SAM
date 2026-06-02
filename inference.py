import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import cfg


SUPPORTED_DATASETS = {'Cornell', 'OCID'}


def numpy_to_torch(array):
    if len(array.shape) == 2:
        return torch.from_numpy(np.expand_dims(array, 0).astype(np.float32))
    return torch.from_numpy(array.astype(np.float32))


def get_structure_classes(dataset_name):
    if dataset_name == 'Cornell':
        from util.data.structure.grasp import GraspMat
        from util.data.structure.img import Image
    elif dataset_name == 'OCID':
        from util.data.structure.grasp_OCID import GraspMat
        from util.data.structure.img_OCID import Image
    else:
        raise ValueError("Available inference datasets: 'Cornell' and 'OCID'.")
    return Image, GraspMat


def load_checkpoint(net, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    net.load_state_dict(state_dict, strict=False)


def save_image_with_point(image, point, box, output_path):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.scatter(point[0], point[1], color='yellow', marker='*', s=120, edgecolors='red')
    rect = plt.Rectangle(
        (box[0], box[1]),
        box[2] - box[0],
        box[3] - box[1],
        linewidth=1,
        edgecolor='red',
        facecolor='none',
    )
    ax.add_patch(rect)
    ax.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def save_image_with_grasp(image, grasps, output_path):
    fig, ax = plt.subplots()
    ax.imshow(image)
    for pose in grasps:
        for i in range(4):
            pt1 = (pose[i][1], pose[i][0])
            pt2 = (pose[(i + 1) % 4][1], pose[(i + 1) % 4][0])
            color = 'red' if i % 2 == 0 else 'green'
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=color)
    ax.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def save_heatmap(able_pred, output_path):
    fig, ax = plt.subplots()
    ax.imshow(able_pred.squeeze().numpy(), cmap='jet', interpolation='nearest')
    ax.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def save_label(label, output_path):
    fig, ax = plt.subplots()
    ax.imshow(label.squeeze().numpy(), cmap='jet')
    ax.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def build_label(args, GraspMat):
    if args.dataset == 'Cornell':
        label_path = args.label_path or args.image_path.replace('r.png', 'grasp.mat')
        return GraspMat(label_path)

    if not args.label_path:
        raise ValueError("OCID inference requires '-label_path /path/to/annotation.txt'.")
    if not args.instance_mask_path:
        raise ValueError("OCID inference requires '-instance_mask_path /path/to/instance_mask.png'.")
    return GraspMat(args.label_path, args.instance_mask_path, 'subset')


def prepare_image_tensor(image, image_size):
    image.nomalise()
    img = cv2.cvtColor(image.img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))
    img = numpy_to_torch(img)
    img = (img - img.min()) / (img.max() - img.min())
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    return transform(TF.to_pil_image(img))


def build_prompts(target, label, args, device):
    from utils import random_box, random_click

    conf_mask = F.interpolate(
        target[:1].unsqueeze(0),
        size=(args.out_size, args.out_size),
        mode='bilinear',
        align_corners=False,
    )
    click_mask_np = cv2.resize(
        target[0].numpy().copy(),
        (args.image_size, args.image_size),
        interpolation=cv2.INTER_LINEAR,
    )

    if click_mask_np.max() != 0:
        point_label, pt = random_click(click_mask_np, 1)
        pt[0], pt[1] = pt[1], pt[0]
    else:
        point_label = -1
        pt = np.array([0, 0], dtype=np.int32)

    pt_coord = pt.copy()
    point_label = torch.tensor([point_label])
    pt = torch.tensor([pt])
    point_coords = torch.as_tensor(pt, dtype=torch.float, device=device)
    labels_torch = torch.as_tensor(point_label, dtype=torch.int, device=device)
    if len(point_label.shape) == 1:
        point_coords = point_coords[:, None, :]
        labels_torch = labels_torch[:, None]
    prompt = (point_coords, labels_torch) if point_label.flatten()[0] != -1 else None

    if args.dataset == 'OCID' and hasattr(label, 'box_mask') and conf_mask.max() != 0:
        box_mask = torch.from_numpy(label.box_mask).unsqueeze(0)
        box_mask = F.interpolate(box_mask, size=(args.image_size, args.image_size), mode='nearest')
    else:
        box_mask = F.interpolate(conf_mask, size=(args.image_size, args.image_size), mode='bilinear')

    box_mask = torch.as_tensor(box_mask > 0.0, dtype=torch.float32)
    x_min, x_max, y_min, y_max = random_box(box_mask)
    box = [y_min, x_min, y_max, x_max]
    combined_box = torch.stack([torch.tensor(box)]).to(dtype=torch.float32, device=device)
    combined_box = combined_box[:, None, :]

    return prompt, combined_box, pt_coord, box


def main():
    args = cfg.parse_args()
    if args.dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Available inference datasets: {sorted(SUPPORTED_DATASETS)}.")
    if not args.image_path:
        raise ValueError("Pass an input image with '-image_path /path/to/image.png'.")
    if not args.pretrain:
        raise ValueError("Pass a trained checkpoint with '-pretrain /path/to/checkpoint.pth'.")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda', args.gpu_device)
    Image, GraspMat = get_structure_classes(args.dataset)
    from utils import get_grasp, get_network

    crop_size = args.crop_size
    if crop_size is None:
        crop_size = 320 if args.dataset == 'Cornell' else 500

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=device, distribution=args.distributed)
    load_checkpoint(net, args.pretrain, device)
    net.eval()

    image = Image(args.image_path)
    label = build_label(args, GraspMat)
    crop_bbox = image.crop(crop_size, args.crop_dist)
    label.crop(crop_bbox)
    label.decode(angle_cls=120)

    img = prepare_image_tensor(image, args.image_size)
    img_in = img.unsqueeze(0).to(device)

    target = numpy_to_torch(label.grasp)
    prompt, combined_box, pt_coord, box = build_prompts(target, label, args, device)

    with torch.no_grad():
        image_embeddings = net.image_encoder(img_in)
        sparse_embeddings, dense_embeddings = net.prompt_encoder(
            points=prompt,
            boxes=combined_box,
            masks=None,
        )
        pred, _ = net.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=net.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=(args.multimask_output > 1),
        )

    pred = F.interpolate(pred, size=(args.out_size, args.out_size), mode='bilinear')
    pred = pred.squeeze(0)
    able_pred = torch.sigmoid(pred[:1])
    angle_pred = torch.sigmoid(pred[1:-1])
    width_pred = torch.sigmoid(pred[-1:])
    grasps = get_grasp(able_pred.unsqueeze(0), angle_pred.unsqueeze(0), width_pred.unsqueeze(0))

    image_np = img_in.squeeze(0).permute(1, 2, 0).cpu().numpy()
    save_heatmap(able_pred.cpu(), os.path.join(args.output_dir, 'heatmap.png'))
    save_image_with_point(image_np, pt_coord, box, os.path.join(args.output_dir, 'prompt.png'))
    save_image_with_grasp(image_np, grasps, os.path.join(args.output_dir, 'grasp.png'))

    label_vis = torch.from_numpy(label.grasp[:1]).unsqueeze(0)
    label_vis = F.interpolate(label_vis, size=(args.image_size, args.image_size), mode='bilinear', align_corners=False)
    save_label(label_vis.squeeze(0), os.path.join(args.output_dir, 'label.png'))


if __name__ == '__main__':
    main()
