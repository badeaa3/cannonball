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
from progressbar import progressbar
import uproot3
from glob import glob
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass


# custom code
sys.path.insert(0,"utils")
sys.path.insert(0,"../utils")
import physics as phy
import logger  as lg
import batcher as bt
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
    parser.add_argument("-m", help="Model you wish to use (masym,drsum)", required=True, type=str)
    parser.add_argument("-d", help="Device to use. If none given then the script will use gpu if it is available, without taking into consideration the gpu's current usage, and cpu if not. ", default=None)
    parser.add_argument("-i", help="Data file to train on.", default=None, required=True)
    parser.add_argument("-o", help="File name of the output", default=None)
    parser.add_argument("-b", help="Batch size", default=2, type=int)
    parser.add_argument("-j", help="Number of workers used by DataLoader. This could significantly affect run time.", default=0, type=int)
    parser.add_argument("-mp", help="Number of cores to use for multiprocessing. If not provided multiprocessing not done.", default=1, type=int)
    parser.add_argument("-ni",  "--ninp",         help="Number of input jets per event",       default=5,    type=int)
    parser.add_argument("-np",  "--nparents",     help="Number of parent particles per event", default=2,    type=int)
    parser.add_argument("-nc",  "--nchildren",    help="Number of children per parent",        default=2,    type=int)
    parser.add_argument("-co",  "--constant",     help="Constant c in Sum |delta R - c|, used with model dr",        default=0,    type=float)
    parser.add_argument("-spt", "--sortbypt",     help="Sort the jets by pt and extract the topk",        action="store_true")
    parser.add_argument("-tk",  "--topk",         help="Top k leading jets to keep after ptsort",         default=5,    type=int)
    parser.add_argument("-pts", "--ptsmearlabel", help="Add a label for ptsmear after the run number. Useful for pt smear of 0.",    default=None, type=float)
    parser.add_argument("-bg",  "--background",  help="The input data files are background files without labels.", action="store_true")
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
    newdir = "%s_%s" % (ops.m,stime)
    if not os.path.isdir(newdir):
        os.mkdir(newdir)
    return newdir

def getmodel():
    ops    = options()
    device = ops.d if ops.d else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if ops.m == "masym":
        from hnet.benchmark.masym import MassAsymmetryHead
        return MassAsymmetryHead(**{"inDim"     : ops.ninp,
                                    "nchildren" : ops.nchildren,
                                    "device"    : device})
    elif ops.m == "drsum":
        from hnet.benchmark.drsum import MinDeltaRSumHead
        return MinDeltaRSumHead(**{"inDim"     : ops.ninp,
                                   "nchildren" : ops.nchildren,
                                   "const"     : ops.constant,
                                   "device"    : device,
                                   "sortbypt"  : ops.sortbypt,
                                   "topk"      : ops.topk})
    else:
        sys.exit("Fatal error: %s" % ("No valid model specified!"))

def announce(model):
    ops = options()
    print("Harvard Stop Event Reconstructor :)")                               
    print("-------------------------------------------")
    print("Benchmark Settings")
    print("Number of input jets per event: %s"       % ops.ninp)
    print("Number of parent particles per event: %s" % ops.nparents)
    print("Number of children per parent: %s"        % ops.nchildren)
    print("------")
    print("Data Settings"                            )
    print("Input file: %s"                           % ops.i)
    print("Save path: %s"                            % output())
    print("Batch size: %i"                           % ops.b)
    print("Number of dataloader workers: %i"         % ops.j)
    print("-------------------------------------------")

def datastruct(newtree=False):
    ''' data structure of output file '''
    ops = options()

    # desired trees
    trees  = [ops.tree]  #["train","test"] if not ops.background else ["bg"]

    # labels
    labels = ["y","p"] if not ops.background else ["p"]

    # quantity to fill (masym or drsum)
    quant, comp = modelquantity()

    # desired branches per tree
    branches    = {}

    # create for each label
    for label in labels:
        # separate branch for each label value
        branches["%s_%s" % (label,quant)] = "float64"
        for i in range(ops.nparents):
            branches["%s_%s%i" % (label,comp,i)] = "float64"
            # for certain models will need to compute the candidate masses
            if ops.m in ["drsum"]:
                branches["%s_m%i" % (label,i)] = "float64"
            # children branches
            for j in range(ops.nchildren):
                branches["%s_m%i_c%i" % (label,i,j)] = "int"
    # branch to label an event train or test
    branches["training_set"]  = "int"
    # use dr sum to check how often ISR in leading 4
    # if ops.m in ["drsum"]:
    #     branches["isrInLeading4"] = "int"
    #     branches["passEtaCut"]    = "int"
    #     branches["passPtCut"]     = "int"

    # make the full data structure
    struct  = {}
    for t in trees:
        struct[t] = {}
        for b,dtype in branches.items():
            struct[t][b] = dtype if newtree else []
    return struct

def modelquantity():
    ops = options()
    if ops.m == "masym":
        return "masym","m"
    elif ops.m == "drsum":
        return "drsum","dr"
    else:
        fatal("No valid model specified!")

def evaluate(config):
    ''' perform the full model evaluation '''

    # handle inputs
    model     = config["model"]
    data      = config["data"]
    save_path = config["save_path"]
    # data, logging
    ops       = options()
    device    = ops.d if ops.d else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dset      = bt.MyDataset(data,ops.b,ops.background)
    loader    = DataLoader(dset, num_workers=ops.j)
    logger    = lg.Logger(**{"nepoch"    : 1,
                             "niter"     : len(dset),
                             "eval_mode" : True})

    # prepare for evaluation
    logger.new_epoch()
    with torch.no_grad():
        for file, data in enumerate(loader):
            # make output data structure
            outdata = datastruct()

            # evaluate on the data
            trainloader,testloader = dset.prepare(data)
            eval(trainloader, model, device, outdata, ops.tree, True)
            eval(testloader, model, device, outdata, ops.tree, False)

            # save to npz and root 
            fname = os.path.basename(dset.data_files[file]).strip(".npz")
            if ops.ptsmearlabel is not None:
                fname = fname.split("_")
                fname.insert(-1,"ptsmear%1.2f" % (ops.ptsmearlabel))
                fname = "_".join(fname)
            oname = "_".join([fname,ops.m])
            oname = os.path.join(save_path, oname)
            np.savez(oname + ".npz", **outdata)

            # update progress bar
            logger.progress(1, file+1)
    return

def eval(data, model, device, outdata, tree, train):
    ''' evaluate model on provided data (X,Y) and populate output data '''
    ops = options()
    for dat in progressbar(data):
        x   = dat[0].to(device)
        p   = model(x)
       
        # populate the labels
        quant, comp = modelquantity()

        # model predictions
        outdata[tree]["p_m0_c0"]         += p[:,0].tolist()
        outdata[tree]["p_m0_c1"]         += p[:,1].tolist()
        outdata[tree]["p_m1_c0"]         += p[:,2].tolist()
        outdata[tree]["p_m1_c1"]         += p[:,3].tolist()
        outdata[tree]["p_%s0" % (comp)]  += p[:,4].tolist()
        outdata[tree]["p_%s1" % (comp)]  += p[:,5].tolist()
        outdata[tree]["p_%s"  % (quant)] += p[:,6].tolist()

        # truth values
        if not ops.background:
            y   = dat[1].to(device)
            nsig = [ops.nchildren] * ops.nparents
            ym,yc = phy.invm(x,y,nsig)
            outdata[tree]["y_m1"]            += ym[:,0].tolist()
            outdata[tree]["y_m0"]            += ym[:,1].tolist()
            outdata[tree]["y_m1_c0"]         += yc[:,0].tolist()
            outdata[tree]["y_m1_c1"]         += yc[:,1].tolist()
            outdata[tree]["y_m0_c0"]         += yc[:,2].tolist()
            outdata[tree]["y_m0_c1"]         += yc[:,3].tolist()
        
            if ops.m in ["masym"]:
                outdata[tree]["y_masym"]     += model.masym.asym(ym).tolist()

        if ops.m in ["drsum"]:
            # get candidate stop masses
            pm = phy.invmBenchmark(x,p)
            outdata[tree]["p_m0"] += pm[:,0].tolist()
            outdata[tree]["p_m1"] += pm[:,1].tolist()

            # get truth delta r information, first 4 of unshuffled are always signal
            if not ops.background:
                ydr = phy.drsum(model,x)
                outdata[tree]["y_dr0"]   += ydr[:,0].tolist()
                outdata[tree]["y_dr1"]   += ydr[:,1].tolist()
                outdata[tree]["y_drsum"] += ydr[:,2].tolist()

            # get the pt sorted indices
            # if model.dr.ind is not None:
            #     isrInLeading4 = torch.sum(torch.flatten(model.dr.ind,start_dim=1)==4,dim=1)
            #     outdata[tree]["isrInLeading4"] += isrInLeading4.tolist()
            #     outdata[tree]["passEtaCut"] += model.dr.passEtaCut.tolist()
            #     outdata[tree]["passPtCut"] += model.dr.passPtCut.tolist()

        # fill the training set bit
        outdata[tree]["training_set"] += [int(train)] * p.shape[0]

if __name__ == "__main__":
    main()
