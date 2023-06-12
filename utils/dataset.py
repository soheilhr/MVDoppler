"""
Datasets for gaits, identity, location, and velocities
"""

import torch
from torch.utils.data import Dataset
import os
import numpy as np
import h5py

def read_h5_basic(path):
    """Read HDF5 files

    Args:
        path (string): a path of a HDF5 file

    Returns:
        radar_dat: micro-Doppler data with shape (256, 128, 2) as (1, time, micro-Doppler, 2 radar channel)
        des: information for radar data
    """
    hf = h5py.File(path, 'r')
    radar_dat = np.array(hf.get('radar_dat'))
    des = dict(hf.attrs)
    hf.close()
    return radar_dat, des

class RadarDataset(Dataset):
    """
    Dataset of different classifications

    The input-output pairs (radar_dat, label) of the RadarDataset are of the following form:
    radar_dat: radar data with shape (micro-Doppler range, time range, 3). The last dimension is three-channels 
                for RGB image. The one-channel data is repeated three times to generate three channels.
    label: depends on the argument `label_type`

    Args:
        file_list (string): path of the file containing labels and information
        data_dir: path of all radar data
        transform: transform function on `radar_dat` contained in `transform_utils.py`
        target_transform: transform function on `label` contained in `transform_utils.py` 
        label_type: wanted label type. Look function `get_label()` for detail.       
        return_des: if True, returns information of the radar data in addition to the input-output pair
        
    """
    def __init__(self,
                 file_list,
                 data_dir,
                 transform=None,
                 target_transform=None,
                 label_type=None,
                 return_des=False,
                 ):
        self.file_list = file_list
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.label_type = label_type
        self.return_des = return_des

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        fname = self.file_list[idx]
        data_path = os.path.join(self.data_dir, fname + '.h5')

        radar_dat, des = read_h5_basic(data_path) 

        if self.transform:
            radar_dat = self.transform(radar_dat)
            
        if self.target_transform:
            label = self.target_transform(des)
        
            if self.return_des:
                return radar_dat.type(torch.FloatTensor), label.type(torch.FloatTensor), des
            else:
                return radar_dat.type(torch.FloatTensor), label.type(torch.FloatTensor)
        else:
            return radar_dat.type(torch.FloatTensor), des
        