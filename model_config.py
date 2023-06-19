import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import albumentations as albu
torch.hub._validate_https_requests_certificates = False
import ssl
ssl._create_default_https_context = ssl._create_unverified_context








    


 
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
  
    WEIGHT_PATH = r'D:/dataset/weight/横向/PS_no_100_0.72.pth'
    # Configure model training

    SIZE = 256
    CHANNELS = 3
    CLASSES = 4
    ENCODER = 'resnet101'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'softmax'

    PREPROCESS = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # Select model architecture in the following line
    MODEL = smp.PSPNet(encoder_name=ENCODER,
                             encoder_weights=ENCODER_WEIGHTS,
                             in_channels=CHANNELS,
                             classes=CLASSES,
                             activation=ACTIVATION)
    ############################################################################################################
    LOSS = nn.CrossEntropyLoss(weight=torch.tensor([0.3584,
                                                    2.2988,
                                                    2.1195,
                                                    3.2961,
                                                  ]))
   
    
    LOSS.__name__ = 'CrossEntropyLoss'




    METRICS = [smp.utils.metrics.Fscore(threshold=0.5)]
    OPTIMIZER = torch.optim.Adam([dict(params=MODEL.parameters(), lr=0.0001)])
    DEVICE = 'cuda'
    TRAIN_BATCH_SIZE = 8
    VAL_BATCH_SIZE = 1
    EPOCHS = 100

    # Select augmentations
    AUGMENTATIONS = [albu.Transpose(p=0.6),
                        #albu.RandomRotate90(p=0.6),
                        #albu.HorizontalFlip(p=0.6),
                        #albu.VerticalFlip(p=0.6)
                     ]
    
  
