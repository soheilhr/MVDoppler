from torch.utils.data import DataLoader
from mmWave_Process.utils.transform_utils import *
from mmWave_Process.utils.dataset import RadarDataset
from torchvision import transforms
import pandas as pd

def LoadDataset_Multi(args):
    """Do transforms on radar data and labels. Load the data from 2 radar sensors.

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


    ### For text/csv files that only have the fname
    else:    
        df = pd.read_csv(args.result.csv_file_train)
        file_list_train = list(df[df.columns[0]])
        df = pd.read_csv(args.result.csv_file_test)
        file_list_test = list(df[df.columns[0]])
        df = pd.read_csv(args.result.csv_file_valid)
        file_list_valid = list(df[df.columns[0]])

    randomize_start = RandomizeStart_SyncRadar(args.transforms.win_size, args.transforms.time_win_start, args.transforms.sync_idx)
    labelmap = LabelMap(label_type=args.transforms.label_type, ymap=args.transforms.ymap_pattern)
    mean_all = args.transforms.radar_mean
    std_all = args.transforms.radar_std
    
    ### Compose the transforms on labels
    composed_label = transforms.Compose([labelmap, ToOneHot(args.train.num_classes)])

    ### Compose the transforms on train set
    composed = transforms.Compose([
        ToCHWTensor(),
        randomize_start,
        SelectChannel(),
        Normalize(mean1=mean_all, std1=std_all, mean2=mean_all, std2=std_all),
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
    composed_test = transforms.Compose([
        ToCHWTensor(),
        CenterStart_SyncRadar(args.transforms.win_size, args.transforms.sync_idx),
        SelectChannel(),
        Normalize(mean1=mean_all, std1=std_all, mean2=mean_all, std2=std_all),
    ])

    radar_valid = RadarDataset(
                                file_list=file_list_valid,
                                data_dir=args.result.data_dir,
                                transform=composed_test, 
                                target_transform=composed_label,
                                label_type=args.transforms.label_type,
                                return_des=args.result.return_des_valid,
                                )
    radar_test =  RadarDataset(
                                file_list=file_list_test,
                                data_dir=args.result.data_dir, 
                                transform = composed_test, 
                                target_transform = composed_label,
                                label_type=args.transforms.label_type,
                                return_des=args.result.return_des_test,
                             )
    data_train = DataLoader(radar_train, batch_size=args.train.batch_size, shuffle=True, num_workers=args.train.num_workers)
    data_valid = DataLoader(radar_valid, batch_size=args.train.batch_size, shuffle=True, num_workers=args.train.num_workers)
    data_test = DataLoader(radar_test, batch_size=args.train.batch_size, shuffle=True, num_workers=args.train.num_workers)

    return data_train, data_valid, data_test, (len(radar_train), len(radar_valid), len(radar_test))
