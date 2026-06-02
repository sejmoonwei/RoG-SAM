# RoG-SAM

RoG-SAM adapts SAM-style prompt segmentation for robotic grasp detection on Cornell and OCID.

This public repository intentionally contains only the training/inference code needed for:

- Cornell grasp dataset
- OCID grasp dataset

Large assets, checkpoints, generated labels, logs, and unrelated experiment code are not tracked.

## Included Code

- `train.py`: training and validation entry point for Cornell/OCID.
- `cfg.py`: command-line arguments.
- `Cornell_inference.py`: Cornell inference example.
- `OCID_inference.py`: OCID inference example.
- `accracy.py`: Cornell evaluation script.
- `OCID_accuracy.py`: OCID evaluation script.
- `dataset/Cornell.py`, `dataset/OCID.py`: dataset loaders.
- `function.py`, `function_cornell.py`, `function_OCID.py`: training and validation routines.
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

## Not Tracked

The repository excludes:

- checkpoints and pretrained weights
- datasets
- logs and generated outputs
- `.npy`, `.zip`, `.pth`, `.pt`, `.ckpt`
- unrelated datasets or experiments

Some local inference scripts may still contain machine-specific example paths. Update those paths before running on a new machine.

## License

This project inherits the GPL license from the upstream Medical SAM Adapter codebase.
