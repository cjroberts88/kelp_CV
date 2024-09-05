############################ ADJUST MODEL and target and confidence levels ########################
#model_name 
model_name = 'trans_hflip_col_syd_fish_s2'

#Load a pre-trained Model
model_file = 'trans_hflip_col_syd_fish_s2_57'

#set target class: kelp = 1, no kelp = 0 #defines which side of the model you are looking at
target_class = 1

#set true classification (i.e. 'kelp present' or 'kelp absent' photos as scored by human annotator)
photo_class = '1'

#set kelp confidence score upper and lower threshold (cut off points for photos to view)
conf_up = '0.48'
conf_low = '0'

cuda_avail = False  #line 73 also needs adjusting if cuda avail/not avail - not sure how to automate.

#########################################################################################

import numpy as np
import pandas as pd
import math 

import matplotlib.pyplot as plt


import torch
import yaml
from model import CustomResNet18

#pip install grad-cam
from pytorch_grad_cam import GradCAM #pip install grad-cam
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from PIL import Image
#import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, ToTensor, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, Normalize, GaussianBlur

# Load the  model

cfg = yaml.safe_load(open('Output/'+model_name+'/model_states/config.yaml', 'r'))

model = CustomResNet18(cfg['num_classes'])  #loads generic resnet model
state = torch.load('Output/'+model_name+'/'+model_file+'.pt') # loads the current 'state' of my model
model.load_state_dict(state['model']) #links the generic pretrained resnet model to my trained model state.

#model = MyModel()
target_layer = [model.feature_extractor.layer4[-1]]
gradcam = GradCAM(model, target_layer, use_cuda=cuda_avail)

#change the target class (ie. noKelp =0, Kelp=1)
labels = [ClassifierOutputTarget(target_class)]

#read predictions csv file and filter to defined parameters (from top)
model_preds = pd.read_csv('Output/'+model_name+'/'+model_file+'.csv', sep=",", index_col=0)
model_preds_sel = model_preds.query('Kelp == '+photo_class+' and predictions >= '+conf_low+' and predictions <= '+conf_up)


#image load and grad cam function
def grad_grid(image_ID,gradcam,labels):
    #labels = [ClassifierOutputTarget(0)]

    image = Image.open(r'Images/'+image_ID)
    image = image.convert('RGB')
    imsize = cfg['image_size']

    loader = Compose([Resize(imsize), ToTensor()])

    image_tensor = loader(image)
    image_tensor = image_tensor.unsqueeze(0)#.to('cuda')

    mask = gradcam(image_tensor, labels)
    mask = mask.squeeze()
    img_resized = image.resize(mask.shape)

    visualisation = show_cam_on_image(np.array(img_resized)/255, mask, use_rgb=True)

    return visualisation, img_resized

#create multi gradcam plot

ncols = 4
nrows = math.ceil(len(model_preds_sel)/2)
fig_height = (nrows/ncols)*20
counter = 1
plt.figure(figsize=(20,fig_height))
for i, row in model_preds_sel.iterrows():
    image_ID = row['id']
    viz, img = grad_grid(image_ID, gradcam,labels)
    ax=plt.subplot(nrows,ncols,counter)
    plt.imshow(img)
    plt.title("iNat obs: "+row['id'][:-4])
    ax.set_axis_off()
    counter = counter + 1
    ax2=plt.subplot(nrows,ncols,counter)
    plt.imshow(viz)
    plt.title("Conf. score: "+str(row['predictions']))
    ax2.set_axis_off()
    counter = counter + 1

plt.savefig('Output/'+model_name+'/figures/gg_'+model_file+'_TC'+str(target_class)+'_PC'+photo_class+'_Clo'+conf_low+'_Cup'+conf_up+'.png',dpi=300, bbox_inches='tight')

### END ###
print('done')