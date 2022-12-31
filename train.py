import torch
import numpy as np
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from PIL import Image
import time
import os
import matplotlib.pyplot as plt
from collections import OrderedDict
import argparse


parser = argparse.ArgumentParser()
parser.add_argument ('data_dir', help = 'Provide path to image')
parser.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth")
parser.add_argument('--arch', dest='arch', default='vgg16', choices=['vgg16', 'densenet121'])
parser.add_argument('--learning_rate', dest='learning_rate', default='0.001',type=float)
parser.add_argument('--hidden_units', dest='hidden_units', default='512')
parser.add_argument('--epochs', dest='epochs', default='1',type=int)
parser.add_argument('--gpu', action='store', default='gpu')
args = parser.parse_args()
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
path = args.save_dir
arch = args.arch
learning_rate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
gpu = args.gpu


train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])])
val_test_transforms  = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
# TODO: Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
val_dataset = datasets.ImageFolder(valid_dir, transform = val_test_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform = val_test_transforms)
# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size = 64)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size = 64)



def main():
    model = getattr(models, args.arch)(pretrained=True)
        
    for element in model.parameters():
        element.requires_grad = False
    
    if args.arch == "vgg16":
        feature_num = model.classifier[0].in_features
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(feature_num, 2048)),
                                  ('drop', nn.Dropout(p=0.5)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(2048, 128)),
                                  ('output', nn.LogSoftmax(dim=1))]))
    elif args.arch == "densenet121":
        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048, 128)),
                                                  ('drop', nn.Dropout(p=0.6)),
                                                  ('relu', nn.ReLU()),
                                                  ('fc2', nn.Linear(128, 64)),
                                                  ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    criterion = nn.NLLLoss() 
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    class_index = train_dataset.class_to_idx
    train(model, criterion, optimizer,trainloader,valloader,testloader,epochs, gpu)
    model.class_to_idx = class_index
    save_checkpoint(path, model, optimizer, args, classifier)
    
    #training
def train(model,criterion, optimizer,trainloader,valloader,testloader,epochs, gpu):
    steps = 0
    print_every = 10
    for e in range(epochs):
        running_loss = 0
        for i, (inputs, labels) in enumerate(trainloader):
                steps += 1
                
                if gpu == 'gpu':
                    model.cuda()
                    inputs, labels = inputs.to('cuda'), labels.to('cuda') 
                else:
                    model.cpu() 
                optimizer.zero_grad()
                
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if steps % print_every == 0:
                    model.eval()
                    valloss = 0
                    accuracy=0
                    for i, (inputs2,labels2) in enumerate(valloader):
                        optimizer.zero_grad()
                        if gpu == 'gpu':
                            inputs2, labels2 = inputs2.to('cuda') , labels2.to('cuda') 
                            model.to('cuda:0')
                        else:
                            pass 
                        with torch.no_grad():
                            outputs = model.forward(inputs2)
                            valloss = criterion(outputs,labels2)
                            ps = torch.exp(outputs).data
                            equality = (labels2.data == ps.max(1)[1])
                            accuracy += equality.type_as(torch.FloatTensor()).mean()
                    valloss = valloss / len(valloader)
                    accuracy = accuracy /len(valloader)
                    print("Epoch: {}/{}... ".format(e+1, epochs),
                        "Training Loss: {:.4f}".format(running_loss/print_every),
                        "Validation Loss {:.4f}".format(valloss),
                        "Accuracy: {:.4f}".format(accuracy),
                        )
                    running_loss = 0

                                 
if __name__ == "__main__":
    main()