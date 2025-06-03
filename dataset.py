# dataset.py
#
# imports the images from locally stored folder of labeled images of day and night, transfortms 
# them to faciliate the processing, and then splits it for training and testing, and 
# sets-up the dataloarders.

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch 

# transform the images
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# import the data
dataset = datasets.ImageFolder(root='data', transform=transform)
generator = torch.Generator().manual_seed(67)

# Split dataset - considering the small dataset 8/2 split seems reasonable
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator = generator)

# get dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

