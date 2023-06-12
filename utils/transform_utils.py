import numpy as np
import torch
import torch.nn.functional as F
import torchvision

### Radar data Transforms 
class ToCHWTensor(object):
    """Convert numpy array to CHW tensor format.
    """
    def __init__(self):
        pass

    def __call__(self, radar_dat):
        radar_dat = radar_dat.transpose((2,0,1)) # (radar channel 1, doppler, time)(1, 128, 256)
        return torch.from_numpy(radar_dat)

class RandomizeStart(object):
    """Randomly select starting time index of snapshot and crop it to
    network input size.

    Args:
        output_len (int): desired output size of time, should be less
            than snapshot length of time
        time_win_start: snapshot time window starting index
    """

    def __init__(self, output_len, time_win_start):
        self.output_len = output_len
        self.time_win_start = time_win_start

    def __call__(self, radar_dat):
        start_idx_min = self.time_win_start
        assert self.output_len <= radar_dat.shape[2], f"network output size {self.output_len} > radar_dat len"

        start_idx_max = radar_dat.shape[2] - self.output_len # e.g. should be 297-256 = 41
        assert start_idx_min <= start_idx_max, f"large start index {start_idx_min}"
        if start_idx_min == start_idx_max:
            start_idx = 0
        else:
            start_idx =np.random.choice(np.arange(start_idx_min, start_idx_max))
        return radar_dat[..., start_idx:start_idx + self.output_len]

class CenterStart(object):
    """crop time range in the center to network input size.

    Args:
        output_len (int): desired output size of time, should be less
            than snapshot length of time
    """

    def __init__(self, output_len):
        self.output_len = output_len

    def __call__(self, radar_dat):
        start_idx = int((radar_dat.shape[2] - self.output_len)/2)
        return radar_dat[..., start_idx:start_idx + self.output_len]

class CropDoppler(object):
    """Crop micro-Doppler range in center into the network input shape
    """

    def __init__(self, output_len):
        self.output_len = output_len

    def __call__(self, radar_dat):
        assert self.output_len <= radar_dat.shape[1], f"network output size {self.output_len} > radar_dat len"

        pos_len = int(self.output_len/2)
        start = int(radar_dat.shape[1]/2) - pos_len
        return radar_dat[..., start:start+self.output_len, :]

class SelectChannel(object):
    """Select the correct radar and repeat the one-channel tensor into three channels
    """
    def __init__(self, radar_idx=None):
        self.radar_idx=radar_idx

    def __call__(self, radar_dat):
        if self.radar_idx is not None:
            assert self.radar_idx in [0,1], f"radar idx {self.radar_idx} out of range of number of radars"
            return torch.repeat_interleave(radar_dat[self.radar_idx:self.radar_idx+1,...], repeats=3, dim=0)
            # (3RGB, 128 doppler, 256 time)
        else:
            return torch.repeat_interleave(torch.unsqueeze(radar_dat,1), repeats=3, dim=1) 
            # (2radar, 3RGB, 128 doppler, 256 time)
        
# class Clip(object):
#     """Crop micro-Doppler range in center into the network input shape
#     """

#     def __init__(self, degree):
#         self.degree = degree

#     def __call__(self, radar_dat):
#         radar_dat = torch.clamp(radar_dat, min=-self.degree, max=self.degree)
#         print(torch.min(radar_dat), torch.max(radar_dat))
#         return radar_dat
    

### Label Transforms 
class LabelMap(object):
    """
    Remap the labels, e.g. integrate two classes into one class.

    Args:
        label_type (str): label type, acceptable values: any columns of des (eg, pattern for gait, subject for person id) or 'location' or 'velocity'
            e.g.
            1. 'pattern': For gait/hand classification, label is one class in {'normal', 'phone call', 'pockets', 'texting'}
            2. 'subject: For identity classification, label is ingeter in [0,12]
            3. 'location': For locations, label is numpy array [x(float), y(float)]
            4. 'velocity': For velocities, label is numpy array [vx(float), vy(float)]
        
        ymap (Dict{Any: Any}): old classes mapping to new classes for classification tasks (not regression).
            e.g. {0:0, 1:0, 2:1, 3:1} Four old classes map into two new classes.
                {'normal':0, 'phone_call':1, 'pockets':2, 'texting':3}   
    """

    def __init__(self, label_type='pattern', ymap=None):
        self.label_type = label_type
        self.ymap = ymap
        
    def __call__(self, des):
        """
        Args:
        des (Dictionary): the des information for the sample
        """
        if self.label_type == 'location':
            label = []
            for label_name in ('x', 'y'):
                label.append(des[label_name])
            label = np.array(label, dtype=float)
        elif self.label_type == 'velocity':
            label = []
            for label_name in ('vx', 'vy'):
                label.append(des[label_name])
            label = np.array(label, dtype=float)
        elif self.label_type in des.keys():
            label = des[self.label_type]
        else:
            print('Error! Label type {label_type} not in the label dictionary!')

        if self.ymap is not None:
            # TODO: assert the labels exist in ymap keys
            label = self.ymap[label]
        return label


class ToOneHot(object):
    """Change an integer label into one-hot label
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        
    def __call__(self, label):
        return F.one_hot(torch.tensor(label, dtype=torch.int64), self.num_classes)


class Normalize(object):
    """
    Apply z-normalization
    """
    def __init__(self, mean1, std1, mean2, std2):
        self.mean = (mean1, mean2)
        self.std = (std1, std2)
    def __call__(self, radar_dat):
        radar_dat[0] = torchvision.transforms.functional.normalize(radar_dat[0], self.mean[0], self.std[0])
        radar_dat[1] = torchvision.transforms.functional.normalize(radar_dat[1], self.mean[1], self.std[1])
        return radar_dat

class RandomShuffle(object):
    """Randomly shuffle the radar
    p: Probility of shuffling
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, radar_dat):
        rand_val = np.random.random_sample()
        if rand_val > self.p:
            radar_dat = radar_dat
        else:
            radar_dat = radar_dat.flip(dims=(0,))       
        return radar_dat
    

class RandomizeStart_SyncRadar(object):
    """Randomly select starting time index of snapshot (with different sync. time with respect to each radar) 
    and crop it to network input size. 

    Args:
        output_len (int): desired output size of time, should be less
            than snapshot length of time
        time_win_start: snapshot time window starting index
        sync_idx: gap of starting points between the two radars
    """

    def __init__(self, output_len, time_win_start, sync_idx):
        self.output_len = output_len
        self.time_win_start = time_win_start
        self.sync_idx = sync_idx

    def __call__(self, radar_dat):
        start_idx_min = self.time_win_start
        assert self.output_len <= radar_dat.shape[2], f"network output size {self.output_len} > radar_dat len"

        start_idx_max = radar_dat.shape[2] - self.output_len - self.sync_idx
        assert start_idx_min <= start_idx_max, f"large start index {start_idx_min}"
        if start_idx_min == start_idx_max:
            start_idx = 0
        else:
            start_idx =np.random.choice(np.arange(start_idx_min, start_idx_max))
        # sync test
        radar_dat_sync = torch.zeros((radar_dat.size(0),radar_dat.size(1),self.output_len),dtype=radar_dat.dtype)
        radar_dat_sync[0,:,:] = radar_dat[0,:,start_idx + self.sync_idx:start_idx + self.sync_idx + self.output_len]
        radar_dat_sync[1,:,:] = radar_dat[1,:,start_idx:start_idx + self.output_len]
        return radar_dat_sync
    
class CenterStart_SyncRadar(object):
    """Select starting time index of snapshot (with different sync. time with respect to each radar) 
    and crop it to network input size. 

    Args:
        output_len (int): desired output size of time, should be less
            than snapshot length of time
        time_win_start: snapshot time window starting index
        sync_idx: gap of starting points between the two radars
    """

    def __init__(self, output_len, sync_idx):
        self.output_len = output_len
        self.sync_idx = sync_idx

    def __call__(self, radar_dat):
        start_idx = int((radar_dat.shape[2] - self.output_len)/2)
        radar_dat_sync = torch.zeros((radar_dat.size(0),radar_dat.size(1),self.output_len),dtype=radar_dat.dtype)
        radar_dat_sync[0,:,:] = radar_dat[0,:,start_idx + self.sync_idx:start_idx + self.sync_idx + self.output_len]
        radar_dat_sync[1,:,:] = radar_dat[1,:,start_idx:start_idx + self.output_len]
        return radar_dat[..., start_idx:start_idx + self.output_len]