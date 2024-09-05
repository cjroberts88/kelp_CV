
# second step after runing training file - move and rename best model evaluation file (.pt) to main folder before running
#to run change directory to folder containing evaluation.py file
# to run: python evaluation.py

import os
import argparse
import yaml
import glob
from tqdm import trange
import cv2
import torchvision.transforms as transforms
import torch
from PIL import Image
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD

import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt

# let's import our own classes and functions!
from util import init_seed
from dataset import CTDataset
from model import CustomResNet18
from PIL import Image

######################

#update model name, model file and evaluation/deplyment image list

model_name = 'trans_hflip_col_syd_fish_s2' #model name from config file (i.e. model folder name)
model_file = 'trans_hflip_col_syd_fish_s2_57' #model pt file name
evaluation_file = 'val.csv' ## validation images

#validation image list
#'val.csv' # validation file

#test/deployment image list
#'deployment_test_list_55k.csv'  #deployment evaluation file - for deploying model on large number of unseen iNat images.

####################

def create_dataloader(cfg, split='train'):
    '''
        Loads a dataset according to the provided split and wraps it in a
        PyTorch DataLoader object.
    '''
    dataset_instance = CTDataset(cfg, split)        # create an object instance of our CTDataset class

    dataLoader = DataLoader(
            dataset=dataset_instance,
            batch_size=cfg['batch_size'],
            shuffle=True,
            num_workers=cfg['num_workers']
        )
    return dataLoader

####################change this if deploying on new images so you can point to a new image folder inexp_resnet18.yaml #######################

#Saved config file from individual model run
# turn on for model training
cfg = yaml.safe_load(open('Output/'+model_name+'/model_states/config.yaml', 'r'))

#Customisable config file - ie. exp_resnet18.yaml
#Can to use this to deploy on new photos if files are not in the base Images/ folder
#Check exp_resnet18.yaml and dataset.py to ensure transform/weighting settings are same as original model yaml 
#cfg = yaml.safe_load(open('../configs/exp_resnet18.yaml', 'r'))



####################################################


init_seed(cfg.get('seed', None))

    # check if GPU is available
device = cfg['device']
if device != 'cpu' and not torch.cuda.is_available():
    print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
    cfg['device'] = 'cpu'

    # initialize data loaders for training and validation set
    #dl_train = create_dataloader(cfg, split='train')
    #dl_val = create_dataloader(cfg, split='val')

    # initialize model
#model, current_epoch = load_model(cfg)

model = CustomResNet18(cfg['num_classes'])

state = torch.load('Output/'+model_name+'/'+model_file+'.pt')
model.load_state_dict(state['model'])
model.eval()

#image = cv2.imread('/mnt/ssd-cluster/chris/Images/49915538.jpg')
# open method used to open different extension image file
#image_ID = '49915538'
#val_label_file
#'/home/chris/k/mnt/ssd-cluster/chris/val_incomplete_exBad.csv'

# dl_train = create_dataloader(cfg, split='train')
# for image, label in dl_train:
#     prediction = model(image)
#     prediction = prediction.detach().numpy()
#     #pytorch_softmax = nn.Softmax(dim=1)
#     print (softmax(prediction, axis=1))
#     break




# import sys
# sys.exit(0)



val_incomplete_exBad = pd.read_csv('Output/'+evaluation_file, sep=",")

#val_incomplete_exBad = pd.read_csv('/mnt/ssd-cluster/chris/val_incomplete_exBad.csv', sep=",")
#
#print(val_incomplete_exBad)

#for x in range(val_incomplete_exBad[0]):
#    print(val_incomplete_exBad[id])
imsize = cfg['image_size']

#print(imsize)

#loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])
transforms = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])

preds_list = [] #create empty list
img_src = cfg['data_root']

for i,row in val_incomplete_exBad.iterrows():
    #print(i,row['id'])
    image_ID = row['id']
    #print(image_ID)
    
    image_path = os.path.join(img_src, image_ID)
    
    image = Image.open(image_path)
    #image = Image.open(r'img_src'+image_ID)
    image = image.convert('RGB')
    #print (np.array(image).shape)


    image_tensor = transforms(image)
    image_tensor = image_tensor.unsqueeze(0)
    #print(image_tensor.shape)
    #break
#'''
    output = model(image_tensor)
    output = output.squeeze()
    output_np = output.detach().numpy()
    #print (output_np)
    preds = softmax(output_np)
    
    #print(preds)

    
#append preds to list
    #print(preds[1])
    preds_list.append(preds[1])
   
val_incomplete_exBad['predictions'] = preds_list   


val_incomplete_exBad.to_csv('Output/'+model_name+'/'+model_file+'.csv', sep=",")
print('done')
#plt.imshow(image)
#plt.title(str(preds[1]))
#plt.savefig('/home/chris/kelp_CV/kelp_classifier/output/'+image_ID+'.jpg')

#'''
