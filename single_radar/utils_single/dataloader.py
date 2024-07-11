from torch.utils.data import DataLoader
from utils.transform_utils import *
from utils.dataset import RadarDataset
from torchvision import transforms
import pandas as pd

def LoadDataset(args):
    """Do transforms on radar data and labels. Load the data.

    Args:
        args: args configured in Hydra YAML file

    """

    if args.result.select_fold:
        df = pd.read_csv(args.result.file_list)

        if args.result.distraction: 
            df = df.loc[(df['pattern']=='pockets') | (df['pattern']=='texting')]
        
        file_list_train = df.loc[ (df['fold'] != args.result.test_fold) & (~df['val_set'])]
        file_list_train = list(file_list_train['fname'])

        file_list_valid = df.loc[ (df['fold'] != args.result.test_fold) & (df['val_set'])]
        file_list_valid = list(file_list_valid['fname'])

        file_list_test = df.loc[ (df['fold'] == args.result.test_fold)]
        file_list_test = list(file_list_test['fname'])

    randomize_start = RandomizeStart(args.transforms.win_size, args.transforms.time_win_start)
    labelmap = LabelMap(label_type=args.transforms.label_type, ymap=args.transforms.ymap_pattern)
    resize = transforms.Resize((args.transforms.resize_doppler, args.transforms.win_size))

    mean = args.transforms.radar_mean
    std = args.transforms.radar_std

    ### Compose the transforms on labels 
    composed_label = transforms.Compose([labelmap, ToOneHot(args.train.num_classes)])

    ### Compose the transforms on train set
    radar_train_dataset_ls = []
    radar_valid_dataset_ls = []
    radar_test_dataset_ls = []
    for radar_idx in args.transforms.select_radar_idx:
        composed = transforms.Compose([
            ToCHWTensor(),
            randomize_start,
            SelectChannel(radar_idx),
            transforms.Normalize(mean=[mean, mean, mean], std=[std, std, std]),
            resize,
        ])

        radar_train = RadarDataset(
                            file_list=file_list_train,
                            data_dir=args.result.data_dir,
                            transform=composed, 
                            target_transform=composed_label,
                            label_type=args.transforms.label_type,
                            return_des=args.result.return_des_train,
                            )


        ### Compose the transforms on valid and test sets
        composed_val = transforms.Compose([
            ToCHWTensor(),
            CenterStart(args.transforms.win_size),
            SelectChannel(radar_idx),
            transforms.Normalize(mean=[mean, mean, mean], std=[std, std, std]),
            resize,
        ])

        radar_valid = RadarDataset(
                                    file_list=file_list_valid,
                                    data_dir=args.result.data_dir,
                                    transform=composed_val, 
                                    target_transform=composed_label,
                                    label_type=args.transforms.label_type,
                                    return_des=args.result.return_des_valid,
                                    )
        
        composed_test = transforms.Compose([
            ToCHWTensor(),
            CenterStart(args.transforms.win_size),
            SelectChannel(radar_idx),
            transforms.Normalize(mean=[mean, mean, mean], std=[std, std, std]),
        ])
        radar_test =  RadarDataset(
                                    file_list=file_list_test,
                                    data_dir=args.result.data_dir, 
                                    transform = composed_test, 
                                    target_transform = composed_label,
                                    label_type=args.transforms.label_type,
                                    return_des=args.result.return_des_test,
                                    )
        
        radar_train_dataset_ls.append(radar_train)
        radar_valid_dataset_ls.append(radar_valid)
        radar_test_dataset_ls.append(radar_test)


    ## Concatenate one-channel data from two radars
    radar_dataset_train = torch.utils.data.ConcatDataset(radar_train_dataset_ls)
    radar_dataset_valid = torch.utils.data.ConcatDataset(radar_valid_dataset_ls)
    radar_dataset_test = torch.utils.data.ConcatDataset(radar_test_dataset_ls)

    data_train = DataLoader(radar_dataset_train, batch_size=args.train.batch_size, shuffle=args.train.shuffle, num_workers=args.train.num_workers)
    data_valid = DataLoader(radar_dataset_valid, batch_size=args.train.batch_size, shuffle=args.train.shuffle, num_workers=args.train.num_workers)
    data_test = DataLoader(radar_dataset_test, batch_size=args.train.batch_size, shuffle=args.train.shuffle, num_workers=args.train.num_workers)

    return data_train, data_valid, data_test
