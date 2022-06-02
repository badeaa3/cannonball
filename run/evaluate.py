#! /usr/bin/env python

'''
Evaluation scipt for colalola architecture. The user must provide the network architecture, 
weights, device, and input data.   

Run like: python evaluate.py -i trainingDataFile.npz -w savedWeights.npz
'''

# python packages
import torch 
from torch import no_grad
from torch.utils.data import DataLoader
import argparse
import numpy as np
import sys
import os
from datetime import datetime
import uproot
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass


# custom code
sys.path.insert(0,"utils")
sys.path.insert(0,"../utils")
from physics  import invm
from logger   import Logger
from batcher  import MyDataset
sys.path.insert(0,"models")
sys.path.insert(0,"../models")


# useful global variables
stime = datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss")

def main():
    ops       = options() 
    save_path = output()
    model     = getmodel()

    # announce settings
    announce(model)

    # check if multiprocessing should be done
    data_list = handleInput() if ops.mp > 1 else [ops.i]

    # create evaluation job dictionaries
    config  = []
    for data in data_list:
        config.append({"model"     : model,
                       "data"      : data,
                       "save_path" : save_path})

    # launch jobs
    ncpu    = ops.mp
    results = mp.Pool(ncpu).map(evaluate, config)


def options():
    ''' argument parser to handle inputs from the user '''
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-a",  help="Architecture you wish to use (colalola)", default="colalola")
    parser.add_argument("-d",  help="Device to use. If none given then the script will use gpu if it is available, without taking into consideration the gpu's current usage, and cpu if not. ", default=None)
    parser.add_argument("-i",  help="Data file to evaluate on.", default=None, required=True)
    parser.add_argument("-o",  help="Directory to save evaluation output to.", default=".")
    parser.add_argument("-b",  help="Batch size", default=2, type=int)
    parser.add_argument("-j",  help="Number of workers used by DataLoader. This could significantly affect run time.", default=0, type=int)
    parser.add_argument("-mp", help="Number of cores to use for multiprocessing. If not provided multiprocessing not done.", default=1, type=int)
    parser.add_argument("-w",  help="Pretrained weights to evaluate with.", default=None, required=True)
    parser.add_argument("-ns", "--nsig",    help="Number of signal jets per parent in the final state", default=[2,2],    type=int, nargs="+")
    parser.add_argument("-ni", "--nisr",    help="Number of isr jets in the final state",    default=1,    type=int)
    parser.add_argument("-nc", "--ncombos", help="Number of combinations to take in CoLa",   default=5,    type=int)
    parser.add_argument("-hd", "--head",    help="Layer widths of head DNN",                 default=[50], type=int, nargs='+')
    parser.add_argument("-bg", "--background",  help="The input data files are background files without labels.", action="store_true")
    parser.add_argument("-t",  "--tree", help="Tree name for the output root files.", default="combinatoric_solution")
    return parser.parse_args()

def handleInput():
    ops = options()
    data = ops.i
    if os.path.isfile(data) and ".npz" in os.path.basename(data):
        return [data]
    elif os.path.isfile(data) and ".txt" in os.path.basename(data):
        return sorted([line.strip() for line in open(data,"r")])
    elif os.path.isdir(data):
        return sorted(os.listdir(data))
    elif "*" in data:
        return sorted(glob(data))
    return []

def output():
    ''' construct the output directory and save path '''
    ops = options()
    if ops.o:
        return ops.o
    newdir = "eval_%s" % (stime)
    if not os.path.isdir(newdir):
        os.mkdir(newdir)
    return newdir

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
    print("Data Settings"                            )
    print("Input file: %s"                           % ops.i)
    print("Pretrained weights: %s"                   % ops.w)
    print("Save path: %s"                            % output())
    print("Batch size: %i"                           % ops.b)
    print("Number of dataloader workers: %i"         % ops.j)
    print("-------------------------------------------")

def getmodel():
    ops    = options()
    device = ops.d if ops.d else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

def datastruct(noutputs,newtree=False):
    ''' data structure of output file '''
    ops = options()

    # desired trees 
    trees  = [ops.tree]  #["train","test"] if not ops.background else ["bg"]

    # labels
    labels = ["y","p"] if not ops.background else ["p"]

    # desired branches per tree
    branches = {}
    # each label value
    for label in labels:
        for il in range(noutputs):
            branches["%s_%i" % (label,il)] = "int"
        # mass and children per parent
        for ic, nc in enumerate(ops.nsig):
            branches["%s_m%i" % (label,ic)] = "float64"
            # make a branch per child
            for c in range(nc):
                branches["%s_m%i_c%i" % (label,ic,c)] = "int64"
        # which jets are not signal (isr,fsr,etc.)
        for ir in range(ops.nisr):
            branches["%s_ns%i" % (label,ir)] = "int64"
    # branch to label an event train or test
    if not ops.background:
        branches["training_set"] = "int"
        
    # make the full data structure
    struct  = {}
    for t in trees:
        struct[t] = {}
        for b,dtype in branches.items():
            struct[t][b] = dtype if newtree else []
    return struct

def evaluate(config):
    ''' perform the full model evaluation '''

    # handle inputs
    model     = config["model"]
    data      = config["data"]
    save_path = config["save_path"]
    # data, logging, and output name 
    ops       = options()
    wname     = os.path.basename(ops.w).strip(".npz")
    device    = ops.d if ops.d else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dset      = MyDataset(data,ops.b,ops.background)
    loader    = DataLoader(dset, num_workers=ops.j)
    logger    = Logger(**{"nepoch"    : 1,
                          "niter"     : len(dset),
                          "eval_mode" : True})

    # prepare logger and model for evaluation
    logger.new_epoch()
    model.eval()
    with torch.no_grad():
        for file, data in enumerate(loader):
            # make output data structure
            outdata = datastruct((sum(ops.nsig) + ops.nisr) + (sum(ops.nsig) -1))

            # evaluate on the data
            trainloader,testloader = dset.prepare(data)
            eval(trainloader, model, device, outdata, ops.tree, True)
            eval(testloader, model, device, outdata, ops.tree, False)

            # save to npz and root 
            fname = os.path.basename(dset.data_files[file]).strip(".npz")
            oname = "_".join([fname,"eval",ops.a])
            oname = os.path.join(save_path, oname)
            np.savez(oname + ".npz", **outdata)

            # update progress bar
            logger.progress(1, file+1)
    return

def eval(data, model, device, outdata, tree, train):
    ''' evaluate model on provided data (X,Y) and populate output data '''
    ops = options()
    for dat in data:
        x   = dat[0].to(device)
        p   = model(x)

        # handle predicted
        fill_outdata(outdata,tree,train,x,p,"p")

        # handle truth if a signal file
        if not ops.background:
            y = dat[1].to(device)
            fill_outdata(outdata,tree,train,x,y,"y")

def fill_outdata(outdata,tree,train,x,y,label):
    ''' fill the outdata given a model evaluation p or truth y and a label "y" or "p" '''
    ops = options()

    # each label value
    for i in range(y.shape[-1]):
        outdata[tree]["%s_%i" % (label,i)] += y[:,i].tolist()

    # mass and children per parent
    ym,yc = invm(x,y,ops.nsig)

    # which jets are not signal (isr,fsr,etc.)
    for ir in range(ops.nisr):
        outdata[tree]["%s_ns%i" % (label,ir)] += yc[:,sum(ops.nsig)+ir].tolist()

    # mass points
    for ic, nc in enumerate(ops.nsig):
        outdata[tree]["%s_m%i" % (label,ic)] += ym[:,ic].tolist()
        cstart = sum(ops.nsig[:ic]) # where to start in children
        for c in range(nc):
            outdata[tree]["%s_m%i_c%i" % (label,ic,c)] += yc[:,cstart+c].tolist()

    # fill the training set bit once
    if label == "y":
        outdata[tree]["training_set"] += [int(train)] * y.shape[0]

if __name__ == "__main__":
    main()
