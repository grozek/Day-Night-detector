# model.py

# Setup the convolutional neural network that processes the image data
# The conv2d helps detect the patterns in images, and MaxPool2d learns to recognize features even
# if they are located in different part of the image. Standard use of ReLU for non-linearity, and
# Linear function for linearity. We get binary classification of day/night in one neuron. 
# 

import torch.nn as nn
import torch

# Convolutional NN 
class CNN(nn.Module):

    # network initallization
    def __init__(self):
        super().__init__()

        # perform convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # get the classification decision from flattened data
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    # perform convolution and classify the image
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

