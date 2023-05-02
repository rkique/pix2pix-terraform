import torch
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from preprocess import GetDataset
import random

dataset = GetDataset()

composed = transforms.Compose(
    [transforms.Resize(286), 
     transforms.RandomCrop(256)])

for i in range(0, len(dataset), 2):
    #Satellite, Elevation Pair
    pair = [dataset[i]['image'], dataset[i+1]['image']]
    pair = [transforms.ToTensor()(image) for image in pair]

    #Performs a Resize and RandomCrop
    transformed_pair = [composed(image) for image in pair]

    #Performs the same flip for both images
    flip = random.choice([transforms.RandomHorizontalFlip(1), 
     transforms.RandomHorizontalFlip(0)])
    transformed_pair = [flip(image) for image in pair]

    print(transformed_pair[0].shape)
    print(transformed_pair[1].shape)
    print(i)
    #plt.show()

OUTPUT_CHANNELS = 3

def downsample(filters, size, apply_batchnorm=True):
    #3 in_channels, 3 out_channels, 4x4 size, 2 stride
    conv = torch.nn.Conv2d(filters, filters, size, strides=2, padding="same", bias=False)
    torch.nn.init.normal(conv.weight, 0, 0.02)

    #expected 3 channels
    batchnorm = torch.nn.BatchNorm2d(3)
    leakyrelu = torch.nn.LeakyReLU()
    
    def call(x):
        x = conv(x)
        if apply_batchnorm:
            x = batchnorm(x)
        x = leakyrelu(x)
        return x

    return call


def upsample(x, filters, size, apply_dropout=False):
    #3 in_channels, 3 out_channels, 4x4 size, 2 stride
    convT = torch.nn.ConvTranspose2d(filters, filters, size, strides=2, padding="same", bias=False)
    torch.nn.init.normal(convT.weight, 0, 0.02)

    #expected 3 channels
    batchnorm = torch.nn.BatchNorm2d(3)
    relu = torch.nn.ReLU()
    dropout = torch.nn.Dropout()

    def call(x):
        x = convT(x)
        x = batchnorm(x)
        if apply_dropout:
            x = dropout(x)
        x = relu(x)
        return x
    
    return call

#inputs go here

down_model = downsample(None)
up_model = upsample(None)

def Generator():

    inputs = torch.tensor(np.zeros(3,256,256))
    down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128, 4),  # (batch_size, 64, 64, 128)
    downsample(256, 4),  # (batch_size, 32, 32, 256)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
  ]
    up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
  ]
    last = torch.nn.ConvTranspose2d(3, 3, 4, strides=2, padding="same", bias=False)
    tanh = torch.nn.Tanh()
    torch.nn.init.normal(last.weight, 0, 0.02)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)
    x = tanh(x)

    return tf.keras.Model(inputs=inputs, outputs=x)