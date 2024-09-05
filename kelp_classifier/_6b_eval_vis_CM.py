#Import libraries
import os
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve


######################### UPDDATE model name, pt file, on confidence threshold ############################

#name of model file
model_name = 'trans_hflip_col_syd_fish_s2' #name of model (i.e. folder name where pt file located)

#name of model pt file
model_file = 'trans_hflip_col_syd_fish_s2_57'

CM_thresh =  0.92  # to calculate 'best' threshold based on F1 or Fbeta score - first run threshold_calc.py


###########################################################################################################


#config file


model_preds = pd.read_csv('Output/'+model_name+'/'+model_file+'.csv', sep=",", index_col=0)
model_preds


y_true = model_preds['Kelp']



##Confusion matrix display

# select position where precision = recall and return threshold value
#PR_Thresh = PRT_array[np.argmin(abs(PRT_array[:,0]-PRT_array[:,1])),2] 

y_pred = (model_preds['predictions'] > CM_thresh).astype(np.float32)

cm = confusion_matrix(y_true, y_pred, labels=None, sample_weight=None, normalize=None)

disp = ConfusionMatrixDisplay(confusion_matrix=cm) #,display_labels=clf.classes_
disp.plot()
plt.title(model_file, fontdict=None, loc='center', pad=10)

plt.savefig('Output/'+model_name+'/figures/'+model_file+'_CM_'+str(CM_thresh)+'.png', dpi=300, format='png')
plt.close()


