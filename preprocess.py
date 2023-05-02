
import os
import random
import numpy as np
import torch
from skimage import io
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class SatelliteElevationDataset(Dataset):
    '''Satellite Elevation Dataset'''

    def __init__(self, root_dirs, tile_cts, transform=None):
        '''
        Arguments:
            root_dir (list[string]): List of paths to the root directory of the image data
            tile_ct (int): number of 51.2 km squares per side of area from which data was collected
            transform (callable, optional): Optional transform to be applied on a sample
        '''
        self.transform = transform
        self.elevation_imgs = []
        self.satellite_imgs = []
        for root_dir, tile_ct in zip(root_dirs, tile_cts):
            for i in range(0, tile_ct * 256, 256):
                for j in range(0, tile_ct * 256, 256):
                    self.elevation_imgs.append(os.path.join(root_dir + f"{i},{j}c.jpg"))
                    self.satellite_imgs.append(os.path.join(root_dir + f"{i},{j}.jpg"))

    def __len__(self):
        return len(self.elevation_imgs)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            # os.path.join(self.root_dir + self.elevation_imgs[idx])
        elevation_img_name = self.elevation_imgs[idx]
        satellite_img_name = self.satellite_imgs[idx]
        elevation_img = io.imread(elevation_img_name)
        satellite_img = io.imread(satellite_img_name)
        sample = {'elevation': elevation_img, 'satellite': satellite_img}
        if self.transform:
            sample = self.transform(sample)
        return sample

# preprocessing: apply random jittering and mirroring to preprocess the training set
def transform(sample):

    transformation = transforms.Compose(
        [transforms.Resize(286), 
        transforms.RandomCrop(256)])
    flip = random.choice([transforms.RandomHorizontalFlip(0), transforms.RandomHorizontalFlip(1)])

    elevation_img, satellite_img = sample['elevation'], sample['satellite']

    elevation_img = transforms.ToTensor()(elevation_img)
    elevation_img = transformation(elevation_img)
    elevation_img = flip(elevation_img)

    
    satellite_img = transforms.ToTensor()(satellite_img)
    satellite_img = transformation(satellite_img)
    satellite_img = flip(satellite_img)
    
    return {'elevation': elevation_img, 'satellite': satellite_img}

def GetDataset():
    return SatelliteElevationDataset(["data/CALI/", "data/ANDES/"], [12, 12], transform=transform)
