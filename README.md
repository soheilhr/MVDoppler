# MVDoppler: Unleashing the Power of Multi-View Doppler for MicroMotion-Based Gait Classification
MVDoppler is a new large multi-view Doppler dataset together with baseline perception models for micro-motion-based gait analysis and classification. The dataset captures the impact of the subject's walking trajectory and radar's observation angle on the classification performance. Additionally, baseline multi-view data fusion techniques are provided to mitigate these effects. This work demonstrates that sub-second micro-motion snapshots can be sufficient for reliable detection of hand movement patterns and even changes in a pedestrian's walking behavior when distracted by their phone. Overall, this research not only showcases the potential of Doppler-based perception, but also offers valuable solutions to tackle its fundamental challenges.

![image](./figures/classes.svg)

URLs of MVDoppler:
* <a href="https://arxiv.org/"> Paper and appendix [arxiv] #TODO </a>
* <a href="https://mvdoppler.github.io/"> MVDoppler project page </a>
* <a href="https://drive.google.com/drive/folders/1Mde8sfxKl8L0OwG4UVQR7IE5Tg-bSosR"> MVDoppler data </a>


## MVDoppler Dataset
This is the repository for MVDoppler containing the code for:
* PyTorch MVDoppler dataset and data loader
* Single-radar and multi-radar baseline codes for micro-motion-based gait analysis and classification, i.e., two tasks hand detection and distraction detection

We tested our baselines on the following environment:
* Python 3.7.16 
* Ubuntu 18.04  
* CUDA 10.0 

## Notice
[2023-06-14] We released MVDoppler version 1.


## Preparing the Dataset
1. Download the dataset from Google Drive <a href="https://drive.google.com/drive/folders/1Mde8sfxKl8L0OwG4UVQR7IE5Tg-bSosR"> here </a> 

2. Unzip folders `Data` and `Labels_and_metadata`. 
* `Data` has all the snapshots in MVDoppler with structure of 
```
dataset
  ├── normal_00000.h5
  ├── normal_00001.h5
  ...
```

All snapshots are in hdf5 formats, and named as `<class>_<ID>.h5`.

* `Labels_and_metadata` 
The dataset should be arranged in the following structure:
```
Labels_and_metadata
  ├── design_table.csv
  ├── meta_data.json
  ├── test.txt
  ├── train.txt
  ├── val.txt
```

`design_table.csv` includes the metadata for the entire dataset (train, val and test) in camma seperated values (CSV) format.

`train.txt`, `val.txt`, and `test.txt` have the snapshot `fname`s in the train set, validation set, and test set for fold 0, respectively.

Labels in `design_table.csv`:
```
- ID: snapshot ID unique per class 
- exp_fname: source episode name (to be used for cross-episode validation)
- fname: full snapshot name
- pattern: class name
- subject: subject index (between 0 and 12)
- notes: not used 
- sex: subject sex
- age: subject age
- height: subject height (cm)
- signal_mean: snapshot signal intensity magnitud mean value
- signal_sd: snapshot signal intensity standard deviation
- snapshot_idx: snapshot index within the corresponding episode
- x: Location across x-axis in Region of Interest (cross-range direction of Radar0), expected to be between [-5,5] meters 
- y: Location across y-axis in Region of Interest (range direction of Radar0), expected to be between [5,15] meters
- noise_db: noise power (dB)
- frame: snapshot median frame index (used to locate a snapshot within its corresponding episode)
- t: snapshot median time relative to the corresponding episode in seconds
- r0: range as measured by radar0
- r1: range as measured by radar1
- vx: velocity on x-axis
- vy: velocity on y-axis
- snr_db_0: signal-to-noise ratio (SNR) observed by radar0 (dB)
- snr_db_1: signal-to_noise ratio (SNR) observed by radar1 (dB)
- episode_idx: episode index within each class
- fold: fold number (4 folds in total)
- val_set: if TRUE, snapshot is used for validation instead of training
```

`meta_data.json` provides details of radar waveforms and processing paramters. More information can be found in supplementary material in the paper <a href="https://arxiv.org/"> Paper [arxiv] </a>
TODO: do we need to specify? Or can mention in the paper?

## Environment Setup
1. Clone this repoitory

2. Create a conda environment
This codebase uses `python==3.7.9`.
```
conda create --name RadarGait python=3.7.9
conda activate RadarGait
```
The package requirements are in `requirements.txt`.

```
pip install -r requirements.txt
```


## Train the model baselines
1. Change argument configurations (This codebase uses `Hydra`, more on this later in Section Argument configurations).  
  In folder `single_radar/conf/`, there are two configuration files for hand detection 'Hand' and distraction detection 'Distract' tasks. 

* Change `result.file_list` to the directory of the `design_table.csv`
* Change `result.data_dir` to the directory of the dataset
* Change `result.path_des` to the directory that you want to store the test results (models, des information, and test confusion matrices)


2. Run single-radar hand movement classification example.   
  Make sure you mount to the repository directory in python files: `line 12 sys.path.append()`
```
cd single_radar
python run_hand.py
```


## Argument configurations

This codebase uses [Hydra](https://github.com/facebookresearch/hydra) to manage
and configure arguments. Hydra offers more flexibility for running and managing complex configurations, and supports rich hierarchical config structure.

The YAML configuration files are in folder `conf`. So you can have a set of arguments in your YAML file like

```YAML
train: 
  learning_rate: 3e-5

transforms:
  win_size: 128
```
You can also use `args.train.learning_rate` to refer to the relevant arguments in scripts.


## Citation
BibTex Reference:
```
@inproceedings{

}
```

## License
The `MVDoppler` dataset is published under the CC BY-NC-ND License, and all codes are published under the Apache License 2.0.
