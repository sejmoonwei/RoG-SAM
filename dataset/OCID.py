import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2
from utils import random_box, random_click
from util.data.structure.img_OCID import Image
from util.data.structure.grasp_OCID import GraspMat
import torchvision.transforms.functional as TF

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

        rota = 30
        rota = np.random.uniform(-1 * rota, rota)
        image.rotate(rota)
        label.rotate(rota)

        dist = 5
        crop_bbox = image.crop(self.crop_size, dist)
        label.crop(crop_bbox)

        flip = True if np.random.rand() < 0.5 else False
        if flip:
            image.flip()
            label.flip()

        image.color()
        image.nomalise()
        img = cv2.cvtColor(image.img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1))
        label.decode(angle_cls=self.angle_k)
        target = label.grasp

        img = self.numpy_to_torch(img)
        target = self.numpy_to_torch(target)

        newsize = (self.img_size, self.img_size)
        click_mask_np = cv2.resize(target[0].numpy().copy(), newsize, interpolation=cv2.INTER_NEAREST)

        if 'click' in self.prompt and click_mask_np.max() != 0:
            point_label = 1
            box_mask = torch.from_numpy(label.box_mask).unsqueeze(0)
            box_mask = F.interpolate(box_mask, size=(self.img_size, self.img_size), mode="nearest")
            box_mask = torch.as_tensor(box_mask > 0.0, dtype=torch.float32)
            click_mask = box_mask.squeeze().numpy()
            point_label, pt = random_click(click_mask, point_label)
            x = pt[1]
            y = pt[0]
            pt[0] = x
            pt[1] = y
        else:
            point_label = -1
            pt = np.array([0, 0], dtype=np.int32)

        if self.transform:
            state = torch.get_rng_state()
            min_val = img.min()
            max_val = img.max()
            img = (img - min_val) / (max_val - min_val)
            img = self.transform(TF.to_pil_image(img))

            conf_mask = F.interpolate(target[:1].unsqueeze(0), size=(self.mask_size, self.mask_size), mode='nearest',
                                      )

            angle_mask = F.interpolate(target[1:-1].unsqueeze(0), size=(self.mask_size, self.mask_size), mode='nearest',
                                       )

            width_mask = F.interpolate(target[-1:].unsqueeze(0), size=(self.mask_size, self.mask_size), mode='nearest',
                                       )
            torch.set_rng_state(state)

            mask = torch.concat([conf_mask.squeeze(0)
                                    , angle_mask.squeeze(0)
                                    , width_mask.squeeze(0)
                                 ], dim=0)  # 122,512,512

        if 'box' in self.prompt and conf_mask.max() != 0:
            box_mask = torch.from_numpy(label.box_mask).unsqueeze(0)
            box_mask = F.interpolate(box_mask, size=(self.img_size, self.img_size), mode="nearest")
            box_mask = torch.as_tensor(box_mask > 0.0, dtype=torch.float32)
            x_min_cup, x_max_cup, y_min_cup, y_max_cup = random_box(box_mask)
            xmin = y_min_cup
            ymin = x_min_cup
            xmax = y_max_cup
            ymax = x_max_cup
            box_cup = [xmin, ymin, xmax, ymax]


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
