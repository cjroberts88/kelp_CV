import pandas as pd
import numpy as np

import sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve

from contextlib import redirect_stdout

model_name = 'trans_hflip_col_del_syd_fish_s2' #name of model (i.e. folder name where pt file located)

#name of model pt file
model_file = 'trans_hflip_col_del_syd_fish_s2_57_crop_max'

#### List of model files available ####
#unweighted_18
#weighted_16
#trans_flip_rot_norm_30
#trans_flip_norm_43
#trans_flip_rot10_norm_45

####

#config file


model_preds = pd.read_csv('Output/'+model_name+'/'+model_file+'.csv', sep=",", index_col=0)
model_preds

y_true = model_preds['Kelp']

y_scores = model_preds['predictions']

precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
f1_scores = 2*recall*precision/(recall+precision)
print('Best threshold: ', thresholds[np.argmax(f1_scores)])
print('Best F1-Score: ', np.max(f1_scores))


f0_5_scores = (1+pow(0.5,2))*((recall*precision)/((pow(0.5,2)*precision)+recall))
print('Best f0.5 threshold: ', thresholds[np.argmax(f0_5_scores)])
print('Best F0.5-Score: ', np.max(f0_5_scores))

f0_8_scores = (1+pow(0.8,2))*((recall*precision)/((pow(0.8,2)*precision)+recall))
print('Best f0.8 threshold: ', thresholds[np.argmax(f0_8_scores)])
print('Best F0.8-Score: ', np.max(f0_8_scores))



with open('Output/'+model_name+'/'+model_file+'_conf_best.txt', 'w') as f:
    with redirect_stdout(f):
       print(model_name)
       print('F1-Score')
       print('Best threshold: ', thresholds[np.argmax(f1_scores)])
       print('Best F1-Score: ', np.max(f1_scores))
       print('F0.5-Score')
       print('Best f0.5 threshold: ', thresholds[np.argmax(f0_5_scores)])
       print('Best F0.5-Score: ', np.max(f0_5_scores))
#       print('F0.8-Score')
#       print('Best f0.8 threshold: ', thresholds[np.argmax(f0_8_scores)])
#       print('Best F0.8-Score: ', np.max(f0_8_scores))
