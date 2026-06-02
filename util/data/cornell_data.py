import os
import numpy as np
import glob

from .grasp_data import GraspDatasetBase


class CornellDataset(GraspDatasetBase):
    """
    """
    def __init__(self, file_path, data_list, data='train', num=-1, test_mode='image-wise', **kwargs):
        """
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(CornellDataset, self).__init__(**kwargs)

        graspf = []
        if test_mode in ['image-wise', 'object-wise', 'all-wise']:
            train_list_f = os.path.join(file_path, '..',  'train-test', data_list, test_mode + '-' + data + '.txt')
            with open(train_list_f) as f:
                names = f.readlines()
                for name in names:
                    name = name.strip()
                    graspf.append(os.path.join(file_path, name + 'grasp.mat'))
        else:
            raise SystemError('Invalid test mode. Choose from [image-wise, object-wise, all-wise].', test_mode)
        graspf.sort()

        if num < 0:
            self.grasp_files = graspf
        else:
            self.grasp_files = graspf[:num]
        
        if len(self.grasp_files) == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))
