import os
import glob
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2
from utils import random_box, random_click
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as TF

class RandomDatasetSampler(Dataset):
    def __init__(self, dataset1, dataset2, dataset3, dataset4, total_size=3600):
        self.dataset1 = dataset1 #Cornell
        self.dataset2 = dataset2 #OCID
        self.dataset3 = dataset3 #REAL
        # self.dataset4 = dataset4 #jac

        self.total_size = total_size
        self.weight = [0.2, 0.4, 0.4]

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        chosen_dataset = random.choices(
            [self.dataset1, self.dataset2, self.dataset3],
            weights = self.weight,
            k=1
        )[0]

        random_idx = random.randint(0,len(chosen_dataset) - 1)
        return chosen_dataset[random_idx]
