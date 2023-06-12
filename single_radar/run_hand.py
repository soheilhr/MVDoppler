import torch
import sys
import os
# Mounted to the repository path
sys.path.append(os.path.abspath('/workspace/MVDoppler/'))

from single_radar.utils_single.dataloader import LoadDataset
from single_radar.utils_single.result_utils import save_result
from single_radar.utils_single.model import MyMobileNet, MyResNet34, MyEfficientNet
from single_radar.utils_single.trainer import Trainer
import hydra
from omegaconf.dictconfig import DictConfig


@hydra.main(config_path="conf", config_name="config_hand_fold")
def main(args: DictConfig) -> None:
    data_train, data_valid, data_test = LoadDataset(args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if args.train.model == 'MyMobileNet':
        model = MyMobileNet(num_classes=args.train.num_classes)
    elif args.train.model == 'MyResNet34':
        model = MyResNet34(num_classes=args.train.num_classes)
    elif args.train.model == 'MyEfficientNet':
        model = MyEfficientNet(num_classes=args.train.num_classes)

    ### Training
    trainer = Trainer(model=model, 
                      data_train=data_train, 
                      data_valid=data_valid, 
                      data_test=data_test, 
                      args=args, 
                      device=device,
                      )
    trainer.train()

    ### Saving results
    set = data_test    
    set_name = 'test'    
    save_result(
        set=set,
        set_name=set_name,
        device=device,
        model=model,
        args=args,
        )

if __name__ == "__main__":
    main()