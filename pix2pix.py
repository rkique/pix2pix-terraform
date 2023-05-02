import torch
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from preprocess import GetDataset
import random

# dataset = GetDataset()


OUTPUT_CHANNELS = 3

def downsample(in_channels, out_channels, size, apply_batchnorm=True):
    #3 in_channels, 3 out_channels, 4x4 size, 2 stride
    conv = torch.nn.Conv2d(in_channels, out_channels, size, stride=2, padding=1, bias=False)
    torch.nn.init.normal(conv.weight, 0, 0.02)

    #expected 3 channels
    batchnorm = torch.nn.BatchNorm2d(3)
    leakyrelu = torch.nn.LeakyReLU()
    
    def call(x):
        print(x)
        x = conv(x)
        if apply_batchnorm:
            x = batchnorm(x)
        x = leakyrelu(x)
        return x

    return call

test_down = downsample(3, 64, 4, apply_batchnorm=False)
inp = torch.zeros((1,3,256,256))
print("Test: ")
print(test_down(inp).size())

def upsample(filters, size, apply_dropout=False):
    #3 in_channels, 3 out_channels, 4x4 size, 2 stride
    convT = torch.nn.ConvTranspose2d(filters, filters, size, stride=2, bias=False)
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

# down_model = downsample(None)
# up_model = upsample(None)

class Generator(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # inputs = torch.tensor(np.zeros(3,256,256))
        self.down_stack = [
            downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
            downsample(128, 4),  # (batch_size, 64, 64, 128)
            downsample(256, 4),  # (batch_size, 32, 32, 256)
            downsample(512, 4),  # (batch_size, 16, 16, 512)
            downsample(512, 4),  # (batch_size, 8, 8, 512)
            downsample(512, 4),  # (batch_size, 4, 4, 512)
            downsample(512, 4),  # (batch_size, 2, 2, 512)
            downsample(512, 4),  # (batch_size, 1, 1, 512)
        ]

        self.up_stack = [
            upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
            upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
            upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
            upsample(512, 4),  # (batch_size, 16, 16, 1024)
            upsample(256, 4),  # (batch_size, 32, 32, 512)
            upsample(128, 4),  # (batch_size, 64, 64, 256)
            upsample(64, 4),  # (batch_size, 128, 128, 128)
        ]

        self.last = torch.nn.ConvTranspose2d(3, 3, 4, stride=2, padding="same", bias=False)
        self.tanh = torch.nn.Tanh()

    # torch.nn.init.normal(last.weight, 0, 0.02)

    def forward(self, x):
        # Downsampling through the model
        skips = []
        for down in self.down_stack:
            print(f"X with size: {x.size()}")
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            x = torch.cat(x, skip)

        x = self.last(x)
        x = self.tanh(x)

        return x

    # return torch.nn.Model(inputs=inputs, outputs=x)

generator = Generator()

# inputs = torch.tensor(np.zeros((3,256,256)))
# plt.imshow(generator(inputs)[0])



LAMBDA = 100

loss_object = torch.nn.BCELoss()

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(disc_generated_output, torch.ones_like(disc_generated_output))

    # mean absolute error
    l1_loss = torch.mean(torch.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

# Define the Discriminator
class Discriminator(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.down1 = downsample(64, 4, False)
        self.down2 = downsample(128, 4)
        self.down3 = downsample(256, 4)

        self.zero_pad1 = torch.nn.ZeroPad2d(1)
        self.conv = torch.nn.Conv2d(256, 512, 4, stride=1, bias=False)

        self.batchnorm = torch.nn.BatchNorm2d(512)
        self.leaky_relu = torch.nn.LeakyReLU()

        self.zero_pad2 = torch.nn.ZeroPad2d(1)

        self.last = torch.nn.Conv2d(512, 1, 4, stride=1, bias=False)

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        x = self.zero_pad1(x)
        x = self.conv(x)

        x = self.batchnorm(x)
        x = self.leaky_relu(x)

        x = self.zero_pad2(x)

        x = self.last(x)

        return x
    
discriminator = Discriminator()
    
# Define discriminator loss
def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(disc_real_output, torch.ones_like(disc_real_output))

    generated_loss = loss_object(disc_generated_output, torch.zeros_like(disc_generated_output))

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


# Optimizers
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

# Generate images
def generate_images(model, test_input, tar):
    prediction = model(test_input)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()

if __name__ == '__main__':
    print('runnin')
