import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2
from utils import random_box, random_click
from util.data.structure.img import Image
from util.data.structure.grasp import GraspMat
import torchvision.transforms.functional as TF
from .utils_cornell import rescale_bbox, crop




class Cornell(Dataset):
    def __init__(self, args, data_path, transform=None, mode='Training', bbox='gd'):
        self.data_path = os.path.join(data_path, mode)
        self.images = []
        self.masks = []


        for filename in os.listdir(self.data_path):
            if filename.endswith(".png"):
                if 'mask' in filename:
                    self.masks.append(os.path.join(self.data_path, filename))
                else:
                    self.images.append(os.path.join(self.data_path, filename))

        self.output_size = 450
        self.angle_k =120
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
        """
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def get_json_path(self, img_path):
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        parent_dir = os.path.dirname(os.path.dirname(img_path))
        grounding_dir = os.path.join(parent_dir, self.grounding_dir)
        box_json_path = os.path.join(grounding_dir, image_name + '.json')
        return box_json_path

    @staticmethod
    def get_box(box_json_path):
        with open(box_json_path, 'r') as f:
            data = json.load(f)
        for item in data:
            if "box" in item:
                box_tensor = torch.tensor(item["box"])
                return box_tensor
        return None

    def __getitem__(self, index):
        point_label = 1

        img_path = self.images[index]
        label_path = img_path.replace('r.png','grasp.mat')
        box_json_path = self.get_json_path(img_path)

        image = Image(img_path)
        label = GraspMat(label_path)
        box_json = self.get_box(box_json_path)

        scale = np.random.uniform(0.95, 1.05)
        image.rescale(scale)
        label.rescale(scale)
        box_json = rescale_bbox(box_json, scale)

        dist = 3
        crop_bbox = image.crop(self.output_size, dist)
        label.crop(crop_bbox)
        box_json = crop(box_json, crop_bbox)

        image.nomalise()

        img = cv2.cvtColor(image.img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1))
        label.decode(angle_cls=self.angle_k)
        target = label.grasp

        img = self.numpy_to_torch(img)
        target = self.numpy_to_torch(target)

        newsize = (self.img_size, self.img_size)
        click_mask_np = cv2.resize(target[0].numpy().copy(), newsize, interpolation = cv2.INTER_LINEAR)

        if 'click' in self.prompt:
            point_label, pt = random_click(click_mask_np , point_label)
            x = pt[1]
            y = pt[0]
            pt[0] = x
            pt[1] = y
        else:
            pt = np.array([0, 0], dtype=np.int32)

        if self.transform:
            state = torch.get_rng_state()
            min_val = img.min()
            max_val = img.max()
            img = (img - min_val) / (max_val - min_val)
            img = self.transform(TF.to_pil_image(img))

            conf_mask = F.interpolate(target[:1].unsqueeze(0), size=(self.mask_size, self.mask_size), mode='bilinear',
                                     align_corners=False)

            angle_mask = F.interpolate(target[1:-1].unsqueeze(0), size=(self.mask_size, self.mask_size), mode='bilinear',
                                     align_corners=False)

            width_mask = F.interpolate(target[-1:].unsqueeze(0), size=(self.mask_size, self.mask_size), mode='bilinear',
                                      align_corners=False)
            torch.set_rng_state(state)

            mask = torch.concat([conf_mask.squeeze(0)
                                    ,angle_mask.squeeze(0)
                                    , width_mask.squeeze(0)
                                 ], dim=0)

        if 'box' in self.prompt:
            if self.bbox_mode == 'gt':
                box_mask = F.interpolate(conf_mask, size=(self.img_size, self.img_size), mode="bilinear")
                box_mask = torch.as_tensor(box_mask > 0.5, dtype=torch.float32)
                x_min_cup, x_max_cup, y_min_cup, y_max_cup = random_box(box_mask)
                xmin = y_min_cup
                ymin = x_min_cup
                xmax = y_max_cup
                ymax = x_max_cup
                box_cup = [xmin, ymin, xmax, ymax]
            elif self.bbox_mode == 'gd':
                box_json = rescale_bbox(box_json, self.img_size / self.output_size )
                box_cup = box_json.tolist()
        else:
            box_cup = [0, 0, 0, 0]

        image_meta_dict = {'filename_or_obj': img_path.split('/')[-1]}
        return {
            'image': img,
            'label': mask,
            'p_label': point_label,
            'pt': pt,
            'box': box_cup,
            'image_meta_dict': image_meta_dict,
        }
