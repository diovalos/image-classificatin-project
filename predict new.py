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
import json
import random
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', default='checkpoint.pth')
parser.add_argument('--top_k', dest='top_k', default='5',type = int)
parser.add_argument('--filepath', dest='filepath', default='flowers/test/10/image_07090.jpg') 
parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
parser.add_argument('--gpu', default='gpu')

args = parser.parse_args()

top_k= args.top_k
checkpoint = args.checkpoint
filepath = args.filepath
gpu = args.gpu

#---------------------------------------------------------
#loading model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    learning_rate = checkpoint['learning_rate']
    epochs = checkpoint['epochs']
    optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def load_cat_names(filename):
    with open(filename) as f:
        category_names = json.load(f)
    return category_names
#---------------------------------------------------------
model = load_checkpoint(checkpoint)
cat_to_name = load_cat_names(args.category_names)
#---------------------------------------------------------
def main(): 
    probs, classes = predict(filepath, model, top_k)
    labels = [cat_to_name[str(index)] for index in classes]
    probability = probs
    print('File selected: ' + filepath)
    
    print(labels)
    print(probability)
    
    i=0 # this prints out top k classes and probs as according to user 
    while i < len(labels):
        print("{} with a probability of {}".format(labels[i], probability[i]))
        i += 1 

#---------------------------------------------------------

#enuf
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
        
    pil_image  = Image.open(image) 
   
    img_transforms  = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                             std=[0.229, 0.224, 0.225])])
    
    output_image = img_transforms(pil_image)
    
    return output_image

#---------------------------------------------------------
#enuf

def predict(imgpath, model, topk=5):
    ''' Get probability values (indeces) and respective flower classes. 
    '''
   
    img_processed = ((process_image(imgpath)).float()).unsqueeze_(0)
    img_processed = model.forward(img_processed.to('cuda'))
    img_probab = F.softmax(img_processed.data,dim=1) 
    image_probability = np.array(img_probab.topk(topk)[0][0])
    
#    model.cuda()
    if gpu == 'gpu'and torch.cuda.is_available():          
        model.cuda()
    else:
        model.cpu() 
            
    index_to_class = {val: key for key, val in model.class_to_idx.items()} 
    classes = [np.int(index_to_class[each]) for each in np.array(img_probab.topk(topk)[1][0])]
    
    return image_probability, classes
#---------------------------------------------------------

if __name__ == "__main__":
    main()