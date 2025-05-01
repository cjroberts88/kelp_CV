
#Need to turn augmentations on/off: Lines 58-65

import os
from torch.utils.data import Dataset
import torch
from torchvision.transforms import Compose, Resize, ToTensor, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, Normalize, GaussianBlur, ColorJitter
from PIL import Image
import csv
import numpy as np
import random

class delete_square(object):


    def __call__(self, img, pixels=100):
            #Delete random square from image
    
        img = np.array(img)
        
        h,w,channels = np.shape(img)
        
        #Random starting pixel
        rh = random.randint(100,120)
        rw = random.randint(100,120)
    
        sub = round(pixels/2)
        add = pixels-sub
        
            #Boundries for square
        hmin = max(rh-sub,50)
        hmax = min(rh+add,h-50)
        vmin = max(rw-sub,50)
        vmax = min(rw+add,w-50)
        
            # Turn pixel within range black
        img[hmin:hmax,vmin:vmax] = np.array([0,0,0])

        img = Image.fromarray(img)
        return img
##################################

class CTDataset(Dataset):
    
    def __init__(self, cfg, split='train'):
        '''
            Constructor. Here, we collect and index the dataset inputs and
            labels.
        '''
        
     ####################################################################   

        self.data_root = cfg['data_root']
        self.split = split
        if split =='train':
            self.transform = Compose([              # Transforms. Here's where we add data augmentation .
                Resize(cfg['image_size']),        # Required- do not turn off - resize the images to the same dimensions...
                RandomHorizontalFlip(cfg['flip_prob']),
                #RandomVerticalFlip(cfg['flip_prob']),
                #RandomRotation(degrees=cfg['rotation_deg']),
                #GaussianBlur((cfg['blur_kern']), (cfg['blur_sigma'])),
                ColorJitter(brightness=(0.5,1.5),contrast=(1),saturation=(0.5,1.5),hue=(-0.1,0.1)),
                #delete_square(),
                ToTensor()#, Required- do not turn off - converts to torch.Tensor. 
                #Normalize((cfg['norm_mean']), (cfg['norm_std'])) # Normalise colours  
                
            ])
        else:
            self.transform = Compose([              # Transformations for test/val data - Augmentations usually not done on val/test data
                Resize(cfg['image_size']),        # Just resize the images to the same dimensions... 
                ToTensor(),                          # ...and convert them to torch.Tensor. 
            ])

##################################################################
            
        # index data into list
        self.data = []

        self.label_mapping = {}
        global_mapping_idx = 0

        if split == 'train':
            ## in case we have multiple train files and wish to merge them
            f = open(cfg['train_label_file'], 'r')
            csv_reader = csv.reader(f, delimiter=',')
            next(csv_reader, None)
            for row in csv_reader:
                self.data.append([row[0], int(row[1])])
            f.close()
                
        elif split=='val':
            f = open(cfg['val_label_file'], 'r')   #insert validate file names in config file. 
        elif split=='test':
            f = open(cfg['test_label_file'], 'r')  #insert test file names in config file. 
        
        
        if split != 'train':
            csv_reader = csv.reader(f, delimiter=',')
            next(csv_reader, None)
            for row in csv_reader:
                self.data.append([row[0], int(row[1])])
                
    def __len__(self):
        '''
            Returns the length of the dataset.
        '''
        return len(self.data)

    
    def __getitem__(self, idx):
        '''
            Returns a single data point at given idx.
            Here's where we actually load the image.
        '''
        if self.split!='unlabeled':
            image_name, label = self.data[idx]              # see line 57 above where we added these two items to the self.data list
        else:
            image_name = self.data[idx]              # see line 57 above where we added these two items to the self.data list
            label = image_name ## this is so we can store predictions in a csv with the image name it came from
        # load image
        image_path = os.path.join(self.data_root, image_name)
        img = Image.open(image_path).convert('RGB')     # the ".convert" makes sure we always get three bands in Red, Green, Blue order

        # transform: see lines  above where we define our transformations
        img_tensor = self.transform(img)
        
        
        return img_tensor, label
        

