import os
import sys
import torch

sys.path.append(os.path.abspath('/workspace/'))
from mmWave_Process.multi_radar.utils_multi.model import FusionMobileNet
from mmWave_Process.multi_radar.utils_multi.trainer import Trainer
from mmWave_Process.multi_radar.utils_multi.result_utils import save_result
from mmWave_Process.multi_radar.utils_multi.dataloader_multi import LoadDataset_Multi

import hydra
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="config_distraction_fold")
def main(args: DictConfig) -> None:
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  data_train, data_valid, data_test, len_data = LoadDataset_Multi(args)
  model = FusionMobileNet(args).to(device)

  # Learning
  trainer = Trainer(model=model, 
                    data_train=data_train, 
                    data_valid=data_valid, 
                    data_test=data_test, 
                    args=args, 
                    device=device,
                    )
  trainer.train()

  cm_set = (data_valid, data_test)
  cm_set_names = ('valid', 'test')
  for set, set_name in zip(cm_set, cm_set_names):
    save_result(
    set=set,
    set_name=set_name,
    device=device,
    model=model,
    args=args,
    )

if __name__ == '__main__':
  main()