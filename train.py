import torch.nn as nn
import torch
from model import SimpleCNN
import torch.nn.functional as F
import dataset
import argparse
import showImage


# TRAIN SETUP

# CREATE PARSER FOR ARGS
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=16)
args = parser.parse_args()


net = SimpleCNN()
optimizer = torch.optim.Adam(net.parameters(),eps=0.000001, lr=0.01, betas=(0.5,0.999), weight_decay=0.0001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
predicted = []
all_preds = []

def train(args, net, device, train_loader, optimizer):
    net.train()  # set model to training mode
    predicted = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # clear old gradients
        output = net(data)                  # forward pass
        
        # Convert logits to predictions: 0 or 1
        preds = (torch.sigmoid(output) > 0.5).int()  # shape: [batch_size, 1]

        # Add predictions to all_preds
        all_preds.append(preds.cpu())  # keep tensors in list
        # If using BCEWithLogitsLoss, target must be float and output raw
        target = target.float().unsqueeze(1)  # shape: [batch_size, 1]
        loss = F.binary_cross_entropy_with_logits(output, target)
        loss.backward()                     # backpropagate
        optimizer.step()                    # update weights
        print("training")
        print(loss)


def test(args, net, device, test_loader):
    with torch.no_grad(): # suppress updating of gradients
        net.eval() # toggle batch norm, dropout
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            target = target.float().unsqueeze(1)

            loss = F.binary_cross_entropy_with_logits(output, target)
            print(loss)
            print("---TESTING---")
    net.train() # toggle batch norm, dropout back again


for epoch in range(1, args.epochs): # training loop
    train(args, net, device, dataset.train_loader, optimizer)
    test(args, net, device, dataset.test_loader)
    showImage.show_images(dataset.train_loader, predicted)
    #showImage.show_images(dataset.train_loader, predicted)

    # periodically evaluate network on test data
    #if epoch % 10 == 0:
    #    test(args, net, device, dataset.test_loader)


