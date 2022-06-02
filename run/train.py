#! /usr/bin/env python

'''
Training scipt for colalola architecture. The user must define the accuracy, loss, training, and validation functions as well as update the ops dictionary. 
It is then left to the user to optimize the network for the dataset.  

Run like: python train.py -i trainingDataFile.npz
'''

# python packages
import torch
from torch.nn.functional import binary_cross_entropy
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
from time import time
from datetime import datetime
import argparse

# custom code
sys.path.insert(0,"utils")
sys.path.insert(0,"../utils")
from logger   import Logger
from batcher  import MyDataset
sys.path.insert(0,"models")
sys.path.insert(0,"../models")

# useful global variables
stime = datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss")

def main():
    ops       = options() 
    device    = ops.d if ops.d else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    save_path = output()
    dset      = MyDataset(ops.i,ops.b)
    loader    = DataLoader(dset, num_workers=ops.j)
    model     = getmodel(device)
    logger    = Logger(**{"logs"    : True, 
                          "warnings": True, 
                          "errors"  : True,
                          "hooks"   : ops.l,
                          "model"   : model,
                          "nepoch"  : ops.e,
                          "niter"   : len(dset),
                          "earlystop_epsilon"              : ops.earlystop_epsilon,
                          "earlystop_lookback"             : ops.earlystop_lookback,
                          "earlystop_within_epoch_epsilon" : ops.earlystop_within_epoch_epsilon,
                          "earlystop_patience"             : ops.earlystop_patience})
    optimizer = torch.optim.Adam(model.parameters(),lr=ops.lr)
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.98)
    # my_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=1)

    # announce settings
    announce(model)

    # perform training
    trainmodel(**{"model"     :model, 
                  "dset"      :dset,
                  "loader"    :loader,
                  "optimizer" :optimizer, 
                  "lr_scheduler": my_lr_scheduler,
                  "logger"    :logger, 
                  "device"    :device, 
                  "nsig"      :sum(ops.nsig),
                  "epochs"    :ops.e})

    # save the logging history and model
    logger.save_history(save_path.replace(".npz", "_logger.npz"))
    torch.save(model.state_dict(), save_path)

def options():
    ''' argument parser to handle inputs from the user '''
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-a", help="Architecture you wish to use (cannonball)", default="cannonball")
    parser.add_argument("-d", help="Device to use. If none given then the script will use gpu if it is available, without taking into consideration the gpu's current usage, and cpu if not. ", default=None)
    parser.add_argument("-i", help="Data file to train on.", default=None, required=True)
    parser.add_argument("-o", help="File name of the output", default=None)
    parser.add_argument("-e", help="Number of epochs to train on", default=1, type=int)
    parser.add_argument("-b", help="Batch size", default=2, type=int)
    parser.add_argument("-lr", help="Learning rate", default=10**-3, type=float)
    parser.add_argument("-j", help="Number of workers used by DataLoader. This could significantly affect run time. ", default=0, type=int)
    parser.add_argument("-w", help="Pretrained weights to use as starting point", default=None)
    parser.add_argument("-l", help="To save the full history of the network during training using the logger.",default=None,)
    parser.add_argument("-ns", "--nsig",    help="Number of signal jets in the final state", default=[2,2],    type=int, nargs="+")
    parser.add_argument("-ni", "--nisr",    help="Number of isr jets in the final state",    default=1,    type=int)
    parser.add_argument("-nc", "--ncombos", help="Number of combinations to take in CoLa",   default=5,    type=int)
    parser.add_argument("-hd", "--head",    help="Layer widths of head DNN",                 default=[50], type=int, nargs='+')
    parser.add_argument("-epe",  "--earlystop_epsilon",              help="Early stopping epsilon for comparing current epoch to previous epochs",   default=10**-3,    type=float)
    parser.add_argument("-epwe", "--earlystop_within_epoch_epsilon", help="Early stopping epsilon for comparing stats within current epoch",         default=10**-3,    type=float)
    parser.add_argument("-epl",  "--earlystop_lookback",             help="Number of epochs to look back to use above epsilon",                      default=5,         type=int)
    parser.add_argument("-epp",  "--earlystop_patience",             help="Number of iterations with an epoch to wait before using above epsilon",   default=30,        type=int)
    return parser.parse_args()

def output():
    ''' construct the output directory and save path '''
    ops = options()
    if not os.path.isdir("save"):
        os.mkdir("save")
    return ops.o if ops.o else os.path.join("save","test_%s.npz" % stime)

def getmodel(device):
    ops = options()
    if ops.a == "colalola":
        from cannonball import Cannonball
        return Cannonball(**{"inDim"      : sum(ops.nsig)  + ops.nisr, 
                           "ncombos"    : ops.ncombos,
                           "noutputs"   : (sum(ops.nsig) + ops.nisr) + (sum(ops.nsig) -1), # isr + decay
                           "finalLayers": ops.head,
                           "device"     : device,
                           "weights"    : ops.w})
    else:
        sys.exit("Fatal error: %s" % ("No valid model specified!"))

def announce(model):
    ops = options()
    print("Cannonball Stop Event Reconstructor :)")                               
    print("-------------------------------------------")
    print("Network Settings")
    print("Number of signal jets in final state: %s" % str(ops.nsig))
    print("Number of isr jets in final state: %i"    % ops.nisr)
    print("Dimension of CoLa: %i"                    % ops.ncombos)
    print("Dimensions of head DNN layers:",            ops.head)
    print("Number of trainable parameters: %i"       % model.count_parameters())
    print("------")
    print("Training Settings"                        )
    print("Number of threads: %i"                    % torch.get_num_threads())
    print("Device: %s"                               % (ops.d if ops.d else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
    print("Number of epochs: %i"                     % ops.e)
    print("------")
    print("Early Stop Settings"                      )
    print("Epsilon: %0.8f"                           % ops.earlystop_epsilon)
    print("Within epoch epsilon: %0.8f"              % ops.earlystop_within_epoch_epsilon)
    print("Lookback: %i"                             % ops.earlystop_lookback)
    print("Patience: %i"                             % ops.earlystop_patience)
    print("------")
    print("Data Settings"                            )
    print("Input file: %s"                           % ops.i)
    print("Pretrained weights: %s"                   % ops.w)
    print("Save path: %s"                            % output())
    print("Batch size: %i"                           % ops.b)
    print("Number of dataloader workers: %i"         % ops.j)
    print("-------------------------------------------")

def trainmodel(model, dset, loader, optimizer, lr_scheduler, logger, device, nsig, epochs):
    ''' perform the full model training '''
    for epoch in range(epochs):
        logger.new_epoch()
        for file, data in enumerate(loader):
            trainloader,testloader = dset.prepare(data)
            train(trainloader, model, optimizer, logger, device, nsig)
            val(testloader, model, logger, device, nsig)
            logger.progress(epoch+1, file+1)
            # if logger.earlystop("mean"):
            #     print("Early stopping condition triggered.")
            #     return
        lr_scheduler.step() #logger.get_avg_stat("val","loss_total"))
    
    # print the model parameters
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)

def train(data, model, optimizer, logger, device, nsig):
    ''' training step given a data (X,Y), model, and optimizer return the loss value and prediction accuracy '''
    # model.train() # Commented out again on October 2nd 2021 while retraining for fix to recalculate the pz component after smearing
    for X,Y in data:
        X,Y = X.to(device),Y.to(device)
        optimizer.zero_grad()
        P = model(X)
        _loss = loss(P, Y, nsig)
        _loss["loss_total"].backward()
        optimizer.step()
        logger.update_stats(step="train",stats=dict(acc(Y,P,nsig),**_loss))
    logger.addtimestamp("train")

def val(data, model, logger, device, nsig):
    ''' validation step given a data (X,Y) and model return the loss value and prediction accuracy '''
    model.eval()
    with torch.no_grad():
        for X,Y in data:
            X,Y = X.to(device),Y.to(device)
            P = model(X)
            _loss = loss(P, Y, nsig)
            logger.update_stats(step="val",stats=dict(acc(Y,P,nsig),**_loss))
    logger.addtimestamp("val")

def acc(y, yhat, nsig, thrs=0.1):
    ''' various accuracy measures for a sigmoid output'''
    
    # convert to numpy for easy calculation
    y_np        = y.cpu().detach().numpy()
    y_isr       = y[:,:-nsig].cpu().detach().numpy()
    y_decay     = y[:,-nsig:].cpu().detach().numpy()
    yhat_np     = yhat.cpu().detach().numpy()
    yhat_isr    = yhat[:,:-nsig].cpu().detach().numpy()
    yhat_decay  = yhat[:,-nsig:].cpu().detach().numpy()
    
    # prepare accuracy dictionary
    acc = {}

    # for each row of P, make the maximum value a 1 and the others a zero. Then check if each row is identical to Y.
    # Return the percentage or equal rows
    x = [i for i in range(y_decay.shape[0])]
    # figure out the index of the jet with the 4th largest ISR check label. Greater than this label set equal to 1, less equal to 0.
    isr   = (yhat_isr   >= yhat_isr[x,np.argpartition(yhat_isr, -4, axis=-1)[:,-4]].reshape(-1,1)).astype(int)
    # set the max value equal to and others to zero 
    decay = (yhat_decay >= yhat_decay[x,np.argmax(yhat_decay,axis=-1)].reshape(-1,1)).astype(int)
    total = np.concatenate([isr,decay],axis=1)
    acc.update({"acc_max_isr"  : np.sum(np.sum(y_isr   == isr,axis=1)   == y_isr.shape[1])   / y_np.shape[0], 
                "acc_max_decay": np.sum(np.sum(y_decay == decay,axis=1) == y_decay.shape[1]) / y_np.shape[0],
                "acc_max_total": np.sum(np.sum(y_np    == total,axis=1) == y_np.shape[1])    / y_np.shape[0]})
    
    # check if each yhat value is with 0.1 of the target y value. if a row has only trues then that row passes.
    # Return the percentage of rows that pass
    acc.update({"acc_thresh_isr"  : np.sum(abs(y_isr   - yhat_isr)   <=thrs)/y_isr.size, 
                "acc_thresh_decay": np.sum(abs(y_decay - yhat_decay) <=thrs)/y_decay.size,
                "acc_thresh_total": np.sum(abs(y_np    - yhat_np)    <=thrs)/y_np.size})
    
    return acc

def loss(P, Y, nsig):
    ''' Custom loss function taken from https://gitlab.cern.ch/atlas-phys-susy-wg/RPVLL/rpvmultijet/alpaca/-/blob/master/bin/alpaca#L127-132 '''
    isr   = binary_cross_entropy(P[:,:-nsig], Y[:,:-nsig])
    decay = binary_cross_entropy(P[:,-nsig:], Y[:,-nsig:])
    return {"loss_isr":isr.detach(),
            "loss_decay":decay.detach(),
            "loss_total":isr+decay}

if __name__ == "__main__":
    main()
