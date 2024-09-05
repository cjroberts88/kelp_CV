'''
    Training script. Here, we load the training and validation datasets (and
    data loaders) and the model and train and validate the model accordingly.

'''
#change directory to folder containing train.py file 
#before running update exp_resnet18.yaml file to include new model name and any image loading transformation parameters(flip probabilty, colour norm values etc)
#before running update dataset.py lines 58-65 to include any image loading transformations (flip, jitter etc)
#to run: python train.py --config ../configs/exp_resnet18.yaml

import os
import argparse
import yaml
import glob
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from sklearn.utils import class_weight

# let's import our own classes and functions!
from util import init_seed
from _2_dataset import CTDataset
from model import CustomResNet18




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
    
    labels_list = []
    for image, label in dataLoader:
        #print(label)
        labels_list.extend(list(label.numpy()))
    #print(labels_list)
    
    #anythingelse = np.array([0,1,1,0,1,0,0,0,0,0,1,0,0,0])

    #weights = class_weight.compute_class_weight('balanced',classes=np.array([0,1]),y=anythingelse)
    labels_list = np.array(labels_list)
    
    #######add class categories to array if adding classes######
    weights = torch.tensor(class_weight.compute_class_weight('balanced',classes=np.array([0,1]),y=labels_list), dtype=torch.float32)
    weights = weights.to(cfg['device'])
    
    #print(weights)

    return dataLoader, weights

def load_model(cfg):
    '''
        Creates a model instance and loads the latest model state weights.
    '''
    model_instance = CustomResNet18(cfg['num_classes'])         # create an object instance of our CustomResNet18 class

    # load latest model state
    model_states = glob.glob(os.path.join('output',cfg['model_name'],'model_states/*.pt'))
    if len(model_states):
        # at least one save state found; get lates
        ####this line doesnt work for some reason### path name not replaced with blank. something to do with the backslashes not matching i think
        model_epochs =   [int(m.replace(os.path.join('output',cfg['model_name'],'model_states/'),'').replace('.pt','')) for m in model_states]
        
        ###use this line instead if restarting model- and manually replace model name...
        #### still doesnt work- graphs fail as early runs not saved
        #model_epochs =   [int(m.replace('trans_hflip_col_incBad_s3\\','') .replace('output\\','').replace('model_states\\','').replace('.pt','')) for m in model_states]
       
        start_epoch = max(model_epochs)

        # load state dict and apply weights to model
        print(f'Resuming from epoch {start_epoch}')
        state = torch.load(open(os.path.join('output',cfg['model_name'],'model_states',f'{start_epoch}.pt'), 'rb'), map_location='cpu')
        model_instance.load_state_dict(state['model'])

    else:
        # no save state found; start anew
        print('Starting new model')
        start_epoch = 0

    return model_instance, start_epoch



def save_model(cfg, epoch, model, stats):
    # make sure save directory exists; create if not
    os.makedirs(os.path.join('output',cfg['model_name'],'model_states'), exist_ok=True)

    # get model parameters and add to stats...
    stats['model'] = model.state_dict()

    # ...and save
    torch.save(stats, open(os.path.join('output',cfg['model_name'],'model_states',f'{epoch}.pt'), 'wb'))
    
    # also save config file if not present
    cfpath = os.path.join('output',cfg['model_name'],'model_states','config.yaml')
    if not os.path.exists(cfpath):
        with open(cfpath, 'w') as f:
            yaml.dump(cfg, f)

            

def setup_optimizer(cfg, model):
    '''
        The optimizer is what applies the gradients to the parameters and makes
        the model learn on the dataset.
    '''
    optimizer = SGD(model.parameters(),
                    lr=cfg['learning_rate'],
                    weight_decay=cfg['weight_decay'])
    return optimizer



def train(cfg, dataLoader, model, optimizer, weights_train):
    '''
        Our actual training function.
    '''

    device = cfg['device']

    # put model on device
    model.to(device)
    
    # put the model into training mode
    # this is required for some layers that behave differently during training
    # and validation (examples: Batch Normalization, Dropout, etc.)
    model.train()

    # loss function
    if cfg['weighting'] == True:
        criterion = nn.CrossEntropyLoss(weights_train)
    else:
        criterion = nn.CrossEntropyLoss()
    # running averages
    loss_total, oa_total = 0.0, 0.0                         # for now, we just log the loss and overall accuracy (OA)

    # iterate over dataLoader
    progressBar = trange(len(dataLoader))
    for idx, (data, labels) in enumerate(dataLoader):       # see the last line of file "dataset.py" where we return the image tensor (data) and label

        # put data and labels on device
        data, labels = data.to(device), labels.to(device)

        # forward pass
        prediction = model(data)

        # reset gradients to zero
        optimizer.zero_grad()

        # loss
        loss = criterion(prediction, labels)

        # backward pass (calculate gradients of current batch)
        loss.backward()

        # apply gradients to model parameters
        optimizer.step()

        # log statistics
        loss_total += loss.item()                       # the .item() command retrieves the value of a single-valued tensor, regardless of its data type and device of tensor

        pred_label = torch.argmax(prediction, dim=1)    # the predicted label is the one at position (class index) with highest predicted value
        oa = torch.mean((pred_label == labels).float()) # OA: number of correct predictions divided by batch size (i.e., average/mean)
        oa_total += oa.item()

        progressBar.set_description(
            '[Train] Loss: {:.2f}; OA: {:.2f}%'.format(
                loss_total/(idx+1),
                100*oa_total/(idx+1)
            )
        )
        progressBar.update(1)
    
    # end of epoch; finalize
    progressBar.close()
    loss_total /= len(dataLoader)           # shorthand notation for: loss_total = loss_total / len(dataLoader)
    oa_total /= len(dataLoader)

    return loss_total, oa_total



def validate(cfg, dataLoader, model, weights_val):
    '''
        Validation function. Note that this looks almost the same as the training
        function, except that we don't use any optimizer or gradient steps.
    '''
    
    device = cfg['device']
    model.to(device)

    # put the model into evaluation mode
    # see lines 103-106 above
    model.eval()
    if cfg['weighting'] == True:
        criterion = nn.CrossEntropyLoss(weights_val)
    else:
        criterion = nn.CrossEntropyLoss()   
       # we still need a criterion to calculate the validation loss

    # running averages
    loss_total, oa_total = 0.0, 0.0     # for now, we just log the loss and overall accuracy (OA)

    # iterate over dataLoader
    progressBar = trange(len(dataLoader))
    
    with torch.no_grad():               # don't calculate intermediate gradient steps: we don't need them, so this saves memory and is faster
        for idx, (data, labels) in enumerate(dataLoader):

            # put data and labels on device
            data, labels = data.to(device), labels.to(device)

            # forward pass
            prediction = model(data)

            # loss
            loss = criterion(prediction, labels)

            # log statistics
            loss_total += loss.item()

            pred_label = torch.argmax(prediction, dim=1)
            oa = torch.mean((pred_label == labels).float())
            oa_total += oa.item()

            progressBar.set_description(
                '[Val ] Loss: {:.2f}; OA: {:.2f}%'.format(
                    loss_total/(idx+1),
                    100*oa_total/(idx+1)
                )
            )
            progressBar.update(1)
    
    # end of epoch; finalize
    progressBar.close()
    loss_total /= len(dataLoader)
    oa_total /= len(dataLoader)

    return loss_total, oa_total



def main():

    # Argument parser for command-line arguments:
    # python ct_classifier/train.py --config configs/exp_resnet18.yaml
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='configs/exp_resnet18.yaml')
    args = parser.parse_args()

    # load config
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))

    # init random number generator seed (set at the start)
    init_seed(cfg.get('seed', None))

    # check if GPU is available
    device = cfg['device']
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        cfg['device'] = 'cpu'

    # initialize data loaders for training and validation set
    dl_train, weights_train = create_dataloader(cfg, split='train')
    dl_val, weights_val = create_dataloader(cfg, split='val')

    # initialize model
    model, current_epoch = load_model(cfg)

    # set up model optimizer
    optim = setup_optimizer(cfg, model)

    # we have everything now: data loaders, model, optimizer; let's do the epochs!
    numEpochs = cfg['num_epochs']

    # make sure save directory exists; create if not
    os.makedirs(os.path.join('output',cfg['model_name'],'figures'), exist_ok=True)

    #storing loss values
    train_losses = []
    train_OAs = []
    val_losses = []
    val_OAs = []

    #best_val_loss = np.inf
    while current_epoch < numEpochs:
        current_epoch += 1
        print(f'Epoch {current_epoch}/{numEpochs}')

        loss_train, oa_train = train(cfg, dl_train, model, optim, weights_train)
        loss_val, oa_val = validate(cfg, dl_val, model, weights_val)

        train_losses.append(loss_train)
        train_OAs.append(oa_train)
        val_losses.append(loss_val)
        val_OAs.append(oa_val)

        # combine stats and save
        stats = {
            'loss_train': loss_train,
            'loss_val': loss_val,
            'oa_train': oa_train,
            'oa_val': oa_val
        }
        save_model(cfg, current_epoch, model, stats)
    
    plt.plot(range(numEpochs), train_losses, label='Train')
    plt.plot(range(numEpochs), val_losses, label='Val')
    plt.title(cfg['model_name'], fontdict=None, loc='center', pad=10)
    plt.legend(loc="lower left")
    plt.xlabel('Number of Epochs')
    plt.ylabel('Losses')

    #plt.savefig(f'/home/chris/kelp_CV/kelp_classifier/output/{cfg["model_name"]}/figures/{cfg["model_name"]}_losses.png', dpi=300, format='png')
    plt.savefig(f'output/{cfg["model_name"]}/figures/{cfg["model_name"]}_losses.png', dpi=300, format='png')
    plt.close()

    plt.plot(range(numEpochs), train_OAs, label='Train')
    plt.plot(range(numEpochs), val_OAs, label='Val')
    plt.title(cfg['model_name'], fontdict=None, loc='center', pad=10)
    plt.legend(loc="lower right")
    plt.xlabel('Number of Epochs')
    plt.ylabel('Overall Accuracy')

    #plt.savefig(f'/home/chris/kelp_CV/kelp_classifier/output/{cfg["model_name"]}/figures/{cfg["model_name"]}_OAs.png', dpi=300, format='png')
    plt.savefig(f'output/{cfg["model_name"]}/figures/{cfg["model_name"]}_OAs.png', dpi=300, format='png')
    plt.close()
    
# That's all, folks!
        


if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python ct_classifier/train.py").
    main()
