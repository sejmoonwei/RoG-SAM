# RoG-SAM: A Language-Driven Framework for Instance-Level Robotic Grasping Detection

Official implementation of:

**RoG-SAM: A Language-Driven Framework for Instance-Level Robotic Grasping Detection**

[Paper](https://ieeexplore.ieee.org/document/10948350)

RoG-SAM adapts SAM-style prompt segmentation for robotic grasp detection on Cornell and OCID.

The current implementation supports:

- Cornell grasp dataset
- OCID grasp dataset

## Included Code

- `train.py`: training and validation entry point for Cornell/OCID.
- `accuracy.py`: unified Cornell/OCID evaluation entry point.
- `inference.py`: unified Cornell/OCID single-image inference and visualization entry point.
- `cfg.py`: command-line arguments.
- `dataset/Cornell.py`, `dataset/OCID.py`: dataset loaders.
- `function.py`: training, validation, and grasp evaluation routines.
- `models/sam/`, `models/ImageEncoder/`, `models/common/`: SAM and adapter model code used by RoG-SAM.
- `util/data/structure/`: grasp/image structures used by Cornell and OCID.

## Environment

```bash
conda env create -f environment.yml
conda activate sam_adapt
```

Download SAM checkpoints separately and place them under `checkpoint/sam/`.

```bash
mkdir -p checkpoint/sam
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
mv sam_vit_b_01ec64.pth checkpoint/sam/
```

## Train

Cornell:

```bash
python train.py \
  -net sam \
  -mod sam_adpt \
  -dataset Cornell \
  -data_path /path/to/cornell_adapt \
  -sam_ckpt checkpoint/sam/sam_vit_b_01ec64.pth \
  -image_size 512 \
  -out_size 512 \
  -b 2
```

OCID:

```bash
python train.py \
  -net sam \
  -mod sam_adpt \
  -dataset OCID \
  -data_path /path/to/OCID_grasp \
  -sam_ckpt checkpoint/sam/sam_vit_b_01ec64.pth \
  -image_size 512 \
  -out_size 512 \
  -b 2
```

## Evaluate

Cornell:

```bash
python accuracy.py \
  -net sam \
  -mod sam_adpt \
  -dataset Cornell \
  -data_path /path/to/cornell_adapt \
  -sam_ckpt checkpoint/sam/sam_vit_b_01ec64.pth \
  -pretrain /path/to/cornell_checkpoint.pth \
  -image_size 512 \
  -out_size 512 \
  -b 1
```

OCID:

```bash
python accuracy.py \
  -net sam \
  -mod sam_adpt \
  -dataset OCID \
  -data_path /path/to/OCID_grasp \
  -sam_ckpt checkpoint/sam/sam_vit_b_01ec64.pth \
  -pretrain /path/to/ocid_checkpoint.pth \
  -image_size 512 \
  -out_size 512 \
  -b 1
```

## Inference

Cornell:

```bash
python inference.py \
  -net sam \
  -dataset Cornell \
  -image_path /path/to/pcd0148r.png \
  -label_path /path/to/pcd0148grasp.mat \
  -sam_ckpt checkpoint/sam/sam_vit_b_01ec64.pth \
  -pretrain /path/to/cornell_checkpoint.pth \
  -image_size 512 \
  -out_size 512 \
  -output_dir outputs/cornell
```

OCID:

```bash
python inference.py \
  -net sam \
  -dataset OCID \
  -image_path /path/to/rgb.png \
  -label_path /path/to/annotation.txt \
  -instance_mask_path /path/to/instance_mask.png \
  -sam_ckpt checkpoint/sam/sam_vit_b_01ec64.pth \
  -pretrain /path/to/ocid_checkpoint.pth \
  -image_size 512 \
  -out_size 512 \
  -output_dir outputs/ocid
```

## Data and Checkpoints

Datasets and pretrained checkpoints should be downloaded separately. The train, evaluation, and inference entry points take data paths and checkpoint paths from command-line arguments.

## License

This project inherits the GPL license from the upstream Medical SAM Adapter codebase.
