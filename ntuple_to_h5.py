'''
convert root to h5 for a p p > g g > t1 t1~ g, (t1 > q q), (t1~ > q q) madgraph sample. 

Use like: 
python ntuple_to_h5.py -i inFile.h5

'''
import os
import uproot3 as uproot
import argparse
import pandas as pd 
import numpy as np
from glob import glob
import multiprocessing as mp
import time
import warnings
import scipy.stats as stats
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=np.VisibleDeprecationWarning)

import time

def main():
    ops     = options()
    ncpu    = ops.j
    data    = input()
    config  = []
    for i in data:
        config.append({"inFile"     : i,
                       "bkg"        : ops.b,
                       "outDir"     : ops.o,
                       "smear"      : ops.smear,
                       "preprocess" : ops.preprocess})
    results = mp.Pool(ncpu).map(process, config)

def options():
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i",           default=None,               help="Input file")
    parser.add_argument("-j",           default=4,                  help="Number of cores to use for multiprocessing.", type=int)
    parser.add_argument("-b",           action='store_true',        help="Flag for background sample. Do not compute training labels.")
    parser.add_argument("-o",           default="./",               help="Output directory for the resulting files")
    parser.add_argument("--smear",      default=None,               help="Smear the pt distribution with a normal distribution (mu,sigma) = (pt,pt*s), where s is the input value.", type=float)
    parser.add_argument("--preprocess", default=None,               help="Apply preprocessing to the data. Options: normToHt", type=str)
    return parser.parse_args()

def input():
    ops = options()
    data = ops.i
    if os.path.isfile(data) and ".root" in os.path.basename(data):
        return [data]
    elif os.path.isfile(data) and ".txt" in os.path.basename(data):
        return sorted([line.strip() for line in open(data,"r")])
    elif os.path.isdir(data):
        return sorted(os.listdir(data))
    elif "*" in data:
        return sorted(glob(data))
    return []

def sampleTillPositive(mu,sigma):
    x = -999
    while x < 0:
        x = np.random.normal(loc=mu,scale=sigma)
    return x

def process(config): 
    branches = {"jet_px"      :"Particle.Px",
                "jet_py"      :"Particle.Py",
                "jet_pz"      :"Particle.Pz",
                "jet_e"       :"Particle.E",
                "jet_pt"      :"Particle.PT",
                "jet_phi"     :"Particle.Phi",
                "jet_eta"     :"Particle.Eta",
                "jet_m"       :"Particle.M",
                "partonindex" :"Particle.PID",
                "mother"      :"Particle.Mother1" # for this process Mother1 = Mother2 always
                }
    for i, [file, df] in enumerate(uproot.pandas.iterate(config["inFile"],"LHEF",reportpath=True,entrysteps=10**5,branches=list(branches.values()))):

        # print file name and prepare output filebase name
        print("Processing file {}".format(file))
        filebase = os.path.basename(file).replace(".root","")

        # rename the df to match standard
        df.rename(columns={v:k for k,v in branches.items()},inplace=True)
        
        # apply particle level cuts
        mask = (df["mother"] == -1) # remove incoming particle
        mask = mask | (abs(df["partonindex"]) >= 1000000) # remove the parent stops, and intermediate sparticles
       
        # reset the subentry index to uniform range
        df = df.where(~mask).dropna().reset_index("subentry",drop=True)
        unique, counts = np.unique(list(df.index), return_counts=True)
        subentry = []
        for i,j in zip(unique,counts):
            subentry += list(range(j))
        df["subentry"] = subentry
        df = df.reset_index()
        df = df.set_index(['entry','subentry'])

        # unstack and save
        df = df.unstack(fill_value=0)

        # If (px,py,pz) will need to be updated after various pre-processing options
        update_cartesian = False
        # smear the pt distribution
        if config["smear"] is not None:
            
            # commented out on October 2nd because I realized it could yield negative pt's and flip (eta,phi)
            # df["jet_pt"]     = df["jet_pt"].applymap(lambda x: np.random.normal(loc=x,scale=x*config["smear"])) 
            
            # commented out truncnorm because it was too slow ~4.8 minutes / 100k events on personal laptop
            # if config["smear"] > 0:
            #     s = time.time()
            #     df["jet_pt"]      = df["jet_pt"].applymap(lambda x: stats.truncnorm(a=-1/config["smear"],    # (0-x)/(eps*x) = -1/eps
            #                                                                         b=(2-1)/config["smear"], # (2*x-x)/(eps*x) = (2-1)/eps
            #                                                                         loc=x,                   # mean
            #                                                                         scale=config["smear"]*x).rvs())      # standard deviation    # to understand a,b formulas look at the notes equation here https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
            #     print("Time: ", time.time() - s)
            # this only takes ~13 seconds / 100k events on personal laptop
            df["jet_pt"]     = df["jet_pt"].applymap(lambda x: sampleTillPositive(x,x*config["smear"])) 
            filebase         += "_ptsmear%1.2f" % (config["smear"])
            update_cartesian = True
        # pre-processing options
        if config["preprocess"] == "normToHt":
            HT = df["jet_pt"].sum(axis=1) # sum of pt 
            df["jet_pt"] = df["jet_pt"].div(HT,axis=0)
            df["jet_e"]  = df["jet_e"].div(HT,axis=0)
            filebase     += "_normToHt"
            update_cartesian = True
        # update the cartesian coordinate branches if necessary
        if update_cartesian:
            df["jet_px"] = df["jet_pt"]*np.cos(df["jet_phi"])
            df["jet_py"] = df["jet_pt"]*np.sin(df["jet_phi"])
            df["jet_pz"] = df["jet_pt"]*np.sinh(df["jet_eta"])
            # added in to enforce that on-shell final state masses are non-negative
            df["jet_e"]  = np.sqrt(df["jet_m"]**2 + df["jet_px"]**2 + df["jet_py"]**2 + df["jet_pz"]**2) 

        # pick branches to save
        df = df[["jet_e","jet_px","jet_py","jet_pz","partonindex","mother"]]

        # save to h5 file
        df.to_hdf(os.path.join(config["outDir"],filebase + ".h5"),
                  key="df",
                  mode='w',
                  append=True) 
        label(df,os.path.join(config["outDir"],filebase+"_shuf0.npz"),
              shuffle_jets=False,
              shuffle_events=False,
              bkg=config["bkg"])
        label(df,os.path.join(config["outDir"],filebase+"_shuf1.npz"),
              shuffle_jets=True,
              shuffle_events=True,
              save_unshuffled_events=True, # save also unshuffled events for easier cross checks
              bkg=config["bkg"])

def label(df,outFileName,shuffle_jets,shuffle_events,save_unshuffled_events=False,bkg=False):
    # some sneaky code to get the df in a np matrix format
    tot_jets_per_event = len(df["partonindex"].columns.get_level_values("subentry"))
    jets_per_event     = 5
    n_variables        = 6 # px,py,pz,e,partonindex,mother
    labeledjets        = np.swapaxes(df.values.reshape(len(df), n_variables, tot_jets_per_event), 1, 2)[:, :jets_per_event, :]
    # pad the jets to get to jets_per_event
    if labeledjets.shape[1] != jets_per_event:
        labeledjets = np.pad(labeledjets,((0,0),(0,jets_per_event-labeledjets.shape[1]),(0,0)))
    # shuffle the jets within each event
    if shuffle_jets:
        for row in labeledjets:
            np.random.shuffle(row)
    # extract the jets and parton indices
    jets = labeledjets[:, :, :4]
    # save and exit if background sample since no need for labels
    if bkg:
        np.savez(outFileName,**{"jets":jets})
        return 0
    # construct the labels
    mother       = np.array(labeledjets[:, :, -1:].squeeze(), dtype=int)
    def find(row): # this is very slow and would be great to speed up
        unique,counts = np.unique(row,return_counts=True)
        return np.isin(row,unique[counts>1])
    stopjet      = np.apply_along_axis(find, 1, mother)
    nonisrlabels = np.array([r.compressed() for r in np.ma.masked_where(stopjet == False, mother)])
    topmatch     = np.array([m == m[0] for m in nonisrlabels]) # 
    labels       = np.concatenate([stopjet, topmatch[:, 1:]], 1)
    if shuffle_events:
        if save_unshuffled_events:
            np.savez(outFileName.replace("_shuf1.npz","_shufjets.npz"),**{"jets":jets,"labels":labels})
        perm     = np.random.permutation(jets.shape[0])
        jets     = jets[perm]
        labels   = labels[perm]
    # save unshuffled to npz
    np.savez(outFileName,**{"jets":jets,"labels":labels})

if __name__ == "__main__":
    main()
