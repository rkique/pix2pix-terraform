import torch
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from preprocess import GetDataset
import random

dataset = GetDataset()

fig = plt.figure()
sample = dataset[67]
elevation_img = sample['elevation']
elevation_img = torch.transpose(elevation_img, 0, 2)
elevation_img = torch.transpose(elevation_img, 1, 0)
plt.imshow(elevation_img)
plt.show()