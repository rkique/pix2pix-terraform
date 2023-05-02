import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class SatelliteElevationDataset(Dataset):

    def __init__(self, root_dir, tile_ct):
        self.root_dir = root_dir 
        self.filenames = []
        for i in range(0, tile_ct * 256, 256):
            for j in range(0, tile_ct * 256, 256):
                self.filenames.append(f"{i},{j}c.jpg")
                self.filenames.append(f"{i},{j}.jpg")

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir + self.filenames[idx])
        image = io.imread(img_name)
        sample = {'image': image}
        return sample

def GetDataset():
    return SatelliteElevationDataset("data/ANDES/", 12)

# print(len(sat_dataset))
# for sample in sat_dataset:
#         print(sample['image'].shape)

# fig = plt.figure()

# for i in range(0,17):
#     ax = plt.subplot(4,5, i+1)
#     plt.tight_layout()
#     sample = sat_dataset[i]
#     plt.imshow(sample['image'])

# plt.show()
