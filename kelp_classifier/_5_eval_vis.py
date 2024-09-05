#Import libraries
import os
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve

########################

#update model name and pt file name
#name of model file
model_name = 'trans_hflip_col_syd_fish_s2' #name of model (i.e. folder name where pt file located)

#name of model pt file
model_file = 'trans_hflip_col_syd_fish_s2_57'

#######################

#config file


model_preds = pd.read_csv('Output/'+model_name+'/'+model_file+'.csv', sep=",", index_col=0)
model_preds

y_true = model_preds['Kelp']

y_scores = model_preds['predictions']
precision, recall, thresholds = precision_recall_curve(
     y_true, y_scores)


##Precision recall curve graph

#calculate average precision score 
# i.e. Area under the curve of precision recall graph

PR_AUC = sklearn.metrics.average_precision_score(y_true, y_scores)

## Add expected performance under a random classifier to precision recall graph

def abline(slope, intercept):
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

kelp_prop = sum(y_true)/len(y_true)
kelp_prop

plot_title = model_name+' (AUC = '+str(round(PR_AUC, 3))+')' #creates plot title string

plt.plot(recall, precision) #precision recall curve
abline(0, kelp_prop) #calling my abline function defined above
plt.title(plot_title, fontdict=None, loc='center', pad=10)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim(0, 1.1)
plt.xlim(0, 1.1)

# make sure save directory exists; create if not
os.makedirs(os.path.join('output',model_name,'figures'), exist_ok=True)

plt.savefig('Output/'+model_name+'/figures/'+model_file+'_PR.png', dpi=300, format='png')
plt.close()
## Threshold vs Recall and Precision graph

thresholds_1 = np.append(thresholds,1)
plt.plot(thresholds_1, recall, label='Recall')
plt.plot(thresholds_1, precision, label='Precision')
plt.title(model_name, fontdict=None, loc='center', pad=10)
plt.legend(loc="lower left")
plt.xlabel('Threshold')
plt.ylabel('Rate')

plt.savefig('Output/'+model_name+'/figures/'+model_file+'_PRT.png', dpi=300, format='png')
plt.close()

##Confusion matrix display

#create precision, recall, threshold dataframe
PRT_array = np.stack([precision,recall,thresholds_1],axis=1) 

# select position where precision = recall and return threshold value
PR_Thresh = PRT_array[np.argmin(abs(PRT_array[:,0]-PRT_array[:,1])),2] 

y_pred = (model_preds['predictions'] > PR_Thresh).astype(np.float32)

cm = confusion_matrix(y_true, y_pred, labels=None, sample_weight=None, normalize=None)

disp = ConfusionMatrixDisplay(confusion_matrix=cm) #,display_labels=clf.classes_
disp.plot()
plt.title(model_file, fontdict=None, loc='center', pad=10)

plt.savefig('Output/'+model_name+'/figures/'+model_file+'_CM.png', dpi=300, format='png')
plt.close()

## Histogram of kelp confidence scores by class

score_hist = model_preds.pivot(columns="Kelp", values="predictions")
plt.hist(score_hist)
plt.title(model_file, fontdict=None, loc='center', pad=10)
plt.xlabel('Confidence score') # from soft max of model kelp scores
plt.ylabel('Frequency')
#plt.show()

plt.savefig('Output/'+model_name+'/figures/'+model_file+'_hist.png', dpi=300, format='png')
plt.close()

### END ###
print('done')
