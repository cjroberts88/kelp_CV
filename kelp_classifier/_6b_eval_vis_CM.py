# to run: python _6b_eval_vis_CM.py

#Import libraries
import os
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve


######################### UPDDATE model name, pt file, on confidence threshold ############################

#name of model file
model_name = 'wtd_aug' #name of model (i.e. folder name where pt file located)

#name of model pt file
model_file = '59' #update to best model pt file name 

#update to perfered threshold
# to calculate 'best' threshold based on F1 or Fbeta score - first run threshold_calc.py
CM_thresh =  0.52 #Update to best threshold based on F1 Score from _6a_threshold_calc  


###########################################################################################################


#config file


model_preds = pd.read_csv('Output/'+model_name+'/'+model_name+'_'+model_file+'.csv', sep=",", index_col=0)
model_preds


y_true = model_preds['Kelp']



##Confusion matrix display

# select position where precision = recall and return threshold value
#PR_Thresh = PRT_array[np.argmin(abs(PRT_array[:,0]-PRT_array[:,1])),2] 

y_pred = (model_preds['predictions'] > CM_thresh).astype(np.float32)

cm = confusion_matrix(y_true, y_pred, labels=None, sample_weight=None, normalize=None)

disp = ConfusionMatrixDisplay(confusion_matrix=cm) #,display_labels=clf.classes_
disp.plot()
plt.title(model_name, fontdict=None, loc='center', pad=10)

plt.savefig('Output/'+model_name+'/figures/'+model_name+'_'+model_file+'_CM_'+str(CM_thresh)+'.png', dpi=300, format='png')
plt.close()


