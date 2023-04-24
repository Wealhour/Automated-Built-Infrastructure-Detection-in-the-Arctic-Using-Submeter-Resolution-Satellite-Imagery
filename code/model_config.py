import segmentation_models_pytorch as smp

import torch
import torch.nn as nn
import albumentations as albu





    


 
class Infra_Config(object):

    # Give the configuration a distinct name related to the experiment
    NAME = None

    # Set paths to data

    # ROOT_DIR = r'/scratch/08968/eliasm1/infra'
    #ROOT_DIR = r'D:/infra-master'
    #WORKER_ROOT =  ROOT_DIR + r'/data/'

    #INPUT_IMG_DIR =  r'D:/dataset/map1'
    #INPUT_MASK_DIR =  r'D:/dataset/mask2'
    TEST_OUTPUT_DIR = r'D:/dataset/test'
    IMG_DIR =  r'D:/dataset/split'
    

   # WEIGHT_PATH = r'D:/dataset/weight/ls6_combined_weighted_2.pth'
  
    WEIGHT_PATH = r'D:/dataset/weight/all_60epoch_0.92_0.87.pth'
    # Configure model training

    SIZE = 256
    CHANNELS = 3
    CLASSES = 4
    ENCODER = 'resnet101'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'softmax'

    PREPROCESS = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # Select model architecture in the following line
    MODEL = smp.UnetPlusPlus(encoder_name=ENCODER,
                             encoder_weights=ENCODER_WEIGHTS,
                             in_channels=CHANNELS,
                             classes=CLASSES,
                             activation=ACTIVATION)

    LOSS = nn.CrossEntropyLoss(weight=torch.tensor([0.2873,
                                                    4.2122,
                                                    5.0599,
                                                    11.7963,
                                                  ]))
   
    
    LOSS.__name__ = 'CrossEntropyLoss'

    METRICS = [smp.utils.metrics.Fscore(threshold=0.5)]
    OPTIMIZER = torch.optim.Adam([dict(params=MODEL.parameters(), lr=0.001)])
    DEVICE = 'cuda'
    TRAIN_BATCH_SIZE = 8
    VAL_BATCH_SIZE = 2
    EPOCHS = 60

    # Select augmentations
    # AUGMENTATIONS = [albu.Transpose(p=0.6),
    #                   albu.RandomRotate90(p=0.6),
    #                   albu.HorizontalFlip(p=0.6),
    #                   albu.VerticalFlip(p=0.6)
    #                 ]
    
    AUGMENTATIONS = [albu.RandomRotate90(p=0.6)]
