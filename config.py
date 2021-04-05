from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()
config.TEST = edict()

## Adam
config.TRAIN.batch_size =10
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

## initialize G
config.TRAIN.n_epoch_init = 1000

## adversarial learning (SRGAN)
config.TRAIN.n_epoch =6000
config.TRAIN.lr_decay = 0.8
config.TRAIN.decay_every = 6000

## train set location
config.TRAIN.CUHK_blur_path = '/Dataset/BlurDetection/CUHK/imgs/'
config.TRAIN.CUHK_gt_path = '/Dataset/BlurDetection/CUHK/GT/'
config.TRAIN.synthetic_blur_path = '/Dataset/BlurDetection/Synthetic_hybrid/imgs/'
config.TRAIN.synyheyic_gt_path = '/Dataset/BlurDetection/Synthetic_hybrid/GT/'


## train image size
config.TRAIN.height = 256#192->144 ->288
config.TRAIN.width = 256 #->288


def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
