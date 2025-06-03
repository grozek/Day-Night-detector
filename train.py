import torch.nn as nn
import torch
from model import CNN
import torch.nn.functional as F
import dataset
import argparse
import showImage
import time

# TRAIN SETUP
# Training on cpu took 275.59 seconds
# Training on mps took 209.16 seconds.


# CREATE PARSER FOR ARGS
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=16)
args = parser.parse_args()

# Varaibles collected used for accuracy visualiation
all_preds_train = []
all_targets_train = []
all_preds_test = []
all_targets_test = []
start_time = time.time()

# Setp
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
net = CNN().to(device)
optimizer = torch.optim.Adam(net.parameters(),eps=0.000001, lr=0.0001, betas=(0.9,0.999), weight_decay=0.00001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)


# TRAIN: train the dataset
def train(args, net, device, train_loader, optimizer, all_preds_train, all_targets_train):
    net.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        print (batch_idx)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()              
        output = net(data)                  
        # Save predictions, and targets. 
        preds = (torch.sigmoid(output) > 0.5).int()
        all_preds_train.extend(preds.squeeze(1).tolist())
        target = target.float().unsqueeze(1)
        all_targets_train.extend(target.squeeze(1).int().tolist())
        # Update loss function, backpropagate, update optimizer
        loss = F.binary_cross_entropy_with_logits(output, target)
        loss.backward()                     
        optimizer.step()                   

    return all_preds_train, all_targets_train

# TEST: test the acccuracy of learnt predictions
def test(args, net, device, test_loader,all_preds_test, all_targets_test):
    with torch.no_grad():
        net.eval()
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            # Save predictions, and targets.
            preds = (torch.sigmoid(output) > 0.5).int()
            all_preds_test.extend(preds.squeeze(1).tolist())
            target = target.float().unsqueeze(1)
            all_targets_test.extend(target.squeeze(1).int().tolist())
            #loss = F.binary_cross_entropy_with_logits(output, target)
    net.train()
    return all_preds_test, all_targets_test

# Loop through all the epochs
for epoch in range(0, args.epochs):
    # Upon every epoch, save the training prediction accuracy
    print(f"Epoch: " + str(epoch))
    all_preds_train, all_targets_train = train(args, net, device, dataset.train_loader, optimizer, all_preds_train, all_targets_train)
    scheduler.step()
# When all learning happened, save the accuracy of test set
all_preds_test, all_targets_test = test(args, net, device, dataset.test_loader, all_preds_test, all_targets_test)

# Record the time of training, show the visualisation of accuracy
end_time = time.time()
print(f"Training on {device} took {end_time - start_time:.2f} seconds.")
showImage.show_images(dataset.train_loader, all_preds_train, all_targets_train, args.epochs, args.batch_size)
showImage.show_images(dataset.test_loader, all_preds_test, all_targets_test, args.epochs, args.batch_size)


# After training is complete
try:
    net.load_state_dict(torch.load("day_night_model.pth", map_location="cpu"))
except Exception as e:
    print("MODEL LOAD FAILED:", e)
