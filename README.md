# Ecklonia radiata (kelp) detection in Underwater Fish Images
Computer vision code developed as part of the [CV4Ecology 2022 Summer School](https://cv4ecology.caltech.edu/).

This repository is code for the computer vision models presented in:
**Trawling for photographic bycatch: Using computer vision to extract kelp presence data from iNaturalist fish photographs**. 


## Training images download

The  training images for this model have not been included in this repository.
They can be downloaded from iNaturalist using the code provided in kelp_classifier/_1_iNat_train_image_download.ipynb
Note: The photographs used for the 'Additional images' are not openly available online due to licencing restrictions. They can be provided directly upon request.


## Model parameter setup
The model parameters can then be setup in the files 
configs/exp_resnet18.yaml
and
kelp_classifier/_2_dataset.py
- #Need to turn augmentations on/off: Lines 58-65

## Model training
Model training can be run using: kelp_classifier/_3_train.py

## Model evaluation

Model training can be run using: kelp_classifier/_4_evaluation.py

 Move and rename best model evaluation file (.pt) to main folder before running

 ## Model performance visualization 
The remaining files are for model performance visualisation:
_5_eval_vis.py

_6a_threshold_calc.py

_6b_eval_vis_CM.py

_7_grad_cam.py
_8_kelp_CV_figures.Rmd

