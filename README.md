# RoG-SAM

RoG-SAM adapts Segment Anything Model (SAM) style prompt-based segmentation for robotic grasp detection experiments. The codebase is based on Medical SAM Adapter and has been extended with grasp datasets, grasp heatmap prediction, inference scripts, and evaluation utilities.

## Supported Datasets

- Cornell
- Jacquard
- OCID
- GraspNet
- REAL
- Mixreal

The original medical segmentation dataset loaders are still present in `dataset/`, but the current project focus is robotic grasping.

## Main Entry Points

- `train.py`: train and validate SAM adapter models.
- `cfg.py`: command-line arguments and defaults.
- `Cornell_inference.py`, `Jacquard_inference.py`, `OCID_inference.py`, `Graspnet_inference.py`, `REAL_inference.py`: dataset-specific inference scripts.
- `*_accuracy.py`: dataset-specific evaluation scripts.
- `graspnet/`: GraspNet rectangle generation and evaluation helpers.
- `util/data/structure/`: grasp and image structures used by inference and evaluation.

## Environment

Create the Conda environment from:

```bash
conda env create -f environment.yml
conda activate sam_adapt
```

There is also a `sam2_environment.yml` file kept for related experiments.

## Checkpoints And Data

Large files are intentionally not tracked in Git:

- model checkpoints: `checkpoint/`
- training logs: `logs/`, `runs/`
- datasets and generated labels
- `.pth`, `.pt`, `.npy`, `.zip`, and similar binary artifacts

Download SAM checkpoints separately and place them under `checkpoint/sam/`, for example:

```bash
mkdir -p checkpoint/sam
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
mv sam_vit_b_01ec64.pth checkpoint/sam/
```

Some scripts currently contain local absolute paths such as `/data/myp/grasp_dataset` or `/data1/samgrasp/...`; update those paths for your machine before training or evaluation.

## Example Training

```bash
python train.py \
  -net sam \
  -mod sam_adpt \
  -dataset Cornell \
  -data_path /path/to/cornell \
  -sam_ckpt checkpoint/sam/sam_vit_b_01ec64.pth \
  -image_size 512 \
  -out_size 512 \
  -b 2
```

For GraspNet, set the camera explicitly when needed:

```bash
python train.py \
  -net sam \
  -mod sam_adpt \
  -dataset Graspnet \
  -camera realsense \
  -sam_ckpt checkpoint/sam/sam_vit_b_01ec64.pth
```

## External Components

`Grounding_sam.py` depends on Grounded Segment Anything / GroundingDINO. That third-party directory and its large weights are not included in this repository; install or clone them separately if you need grounding-based prompts.

## License

This repository inherits the GPL license from the upstream Medical SAM Adapter codebase.
