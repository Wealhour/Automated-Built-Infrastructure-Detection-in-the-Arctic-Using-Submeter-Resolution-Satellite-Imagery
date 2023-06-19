from model_config import *
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import *
from torch.utils.data import DataLoader
from dataloader import *
from tqdm import tqdm
import itertools
from matplotlib.colors import ListedColormap

# Evaluation and Visualization

# load best saved checkpoint

device = torch.device(Infra_Config.DEVICE)
best_model = torch.load(Infra_Config.WEIGHT_PATH)
best_model.to(device)

# Create test dataset for model evaluation and prediction visualization

x_test_dir = Infra_Config.IMG_DIR + '/x_test1'
y_test_dir = Infra_Config.IMG_DIR + '/y_test1'

test_dataset = Dataset(
    x_test_dir, 
    y_test_dir, 
    preprocessing=get_preprocessing(Infra_Config.PREPROCESS),
)

test_dataloader = DataLoader(test_dataset)

test_dataset_vis = Dataset(
    x_test_dir,
    y_test_dir
)

# Evaluate model on test dataset

test_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=Infra_Config.LOSS,
    metrics=Infra_Config.METRICS,
    device=Infra_Config.DEVICE,
)

logs = test_epoch.run(test_dataloader)

# Create function to visualize predictions



# Create function to visualize predictions
def visualize(**images):
    """Plot images in one row."""
    n = len(images)
    plt.figure(figsize=(5, 5))

    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    #plt.show()

# Visualize predictions on test dataset.
for i, id_ in tqdm(enumerate(test_dataset), total=len(test_dataset)):
    
    image_vis = test_dataset_vis[i][0].astype('float')
    image_vis = image_vis/65535

    visualize(
       image=image_vis
       )

    name = Infra_Config.TEST_OUTPUT_DIR + '/test_preds/' + str(i) + '.png'
    plt.savefig(name)


