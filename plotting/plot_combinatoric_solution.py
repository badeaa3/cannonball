"""
Plot the combinatoric solutions by loading the result npz and file evaluated on
"""

# general imports 
import ROOT
import argparse
import os
import sys
import time
import numpy as np
import glob
import multiprocessing as mp
import time

# user made
sys.path.insert(0,"utils")
sys.path.insert(0,"../utils")
import physics as phys

# useful parameters
stime = time.strftime("%Y_%m_%d_%Hh%Mm%Ss")

def main():
    ops = options()
    print(ops)

    # load files and trees
    data_list  = handleInput(ops.d)
    comb_list  = handleInput(ops.c)

    # create evaluation job dictionaries
    config  = []
    for data_file,comb_file in zip(data_list,comb_list):
        config.append({"data_file" : data_file,
                       "comb_file" : comb_file,
                       "out_dir"   : ops.o,
                       "bkg"       : ops.bkg})

    # launch jobs
    ncpu    = ops.mp
    results = mp.Pool(ncpu).map(analyze, config)

def options():
    """ input argument parser """
    parser = argparse.ArgumentParser()
    # all model configurations
    parser.add_argument("-d",  help="Data files evaluated on", required=True)
    parser.add_argument("-c",  help="Combinatoric solution results", required=True)
    parser.add_argument("-o",  help="Output directory for the resulting files", default="./")
    parser.add_argument("-mp", help="Number of cores to use for multiprocessing. If not provided multiprocessing not done.", default=1, type=int)
    parser.add_argument("--bkg", help="If background sample.", action="store_true")
    return parser.parse_args()

def handleInput(data):
    if os.path.isfile(data) and ".npz" in os.path.basename(data):
        return [data]
    elif os.path.isfile(data) and ".txt" in os.path.basename(data):
        return sorted([line.strip() for line in open(data,"r")])
    elif os.path.isdir(data):
        return sorted(os.listdir(data))
    elif "*" in data:
        return sorted(glob.glob(data))
    return []

def mass_map(file):
    masses = {"run_%1.2i" % (iF): ((iF+1) * 100) for iF in range(2,20)}
    return [val for key,val in masses.items() if key in file][0]

def getNp1DBinError(vals,bins,density):
    h,b = np.histogram(vals,bins=bins,density=False)
    if density:
        # normalized bin error = bin error unnormalized / integral of unnormalized bins = sqrt(bin unnormalized) / integral of unnormalized bins
        return np.sqrt(h) / sum(np.diff(b)*h)
    # otherwise just sqrt(bin unnormalized)
    return np.sqrt(h)

def analyze(config):

    # load the data
    data_file = np.load(config["data_file"], allow_pickle=True)
    jets = data_file["jets"]
    
    # load the combinatoric solution
    comb_file = np.load(config["comb_file"], allow_pickle=True)
    if "colalola" in config["comb_file"]:
        comb = comb_file["combinatoric_solution"].item()
    elif any(x in config["comb_file"] for x in ["masym","drsum"]):
        try:
            comb = comb_file["bg"].item() # PRL submission files have this
        except:
            comb = comb_file["combinatoric_solution"].item() # PRD submission files have this
    else:
        print("no valid file provided!")
        exit()
    # convert to np
    for key,val in comb.items():
        if not isinstance(val,np.ndarray):
            comb[key] = np.array(val)   

    # create an ouput dictionary for computed values and histograms
    output = {}

    # if hists should be made with density on
    density = True # only applied to masym, mavg, drsum, cos(theta*)
    fracEvents = False # divide by number of events

    # determine mass point
    if config["bkg"]:
        mass = 1000
        py = ["p"]
    else:
        mass = mass_map(config["data_file"])
        py = ["p","y"]

    # compute variables and histograms
    for i in py:
        # get number of events
        nevents = jets.shape[0]

        # ensure combinatoric solutions are ints in case the model saved as float (ex. 0.0, 1.0)
        comb[f"{i}_m0_c0"] = comb[f"{i}_m0_c0"].astype(int)
        comb[f"{i}_m0_c1"] = comb[f"{i}_m0_c1"].astype(int)
        comb[f"{i}_m1_c0"] = comb[f"{i}_m1_c0"].astype(int)
        comb[f"{i}_m1_c1"] = comb[f"{i}_m1_c1"].astype(int)

        # extract the children 4-momenta in (px,py,pz,e)
        mom4_m0_c0 = jets[np.arange(jets.shape[0]),comb[f"{i}_m0_c0"]]
        mom4_m0_c1 = jets[np.arange(jets.shape[0]),comb[f"{i}_m0_c1"]]
        mom4_m1_c0 = jets[np.arange(jets.shape[0]),comb[f"{i}_m1_c0"]]
        mom4_m1_c1 = jets[np.arange(jets.shape[0]),comb[f"{i}_m1_c1"]]

        # create parent 4-momenta
        mom4_m0 = mom4_m0_c0 + mom4_m0_c1
        mom4_m1 = mom4_m1_c0 + mom4_m1_c1

        # extract ISR exploiting mapping trick from the sum of indices
        where_is_isr = {6:4, # 0,1,2,3 = 6 -> isr is 4
                        7:3, # 0,1,2,4 = 7 -> isr is 3
                        8:2, # 0,1,3,4 = 8 -> isr is 2
                        9:1, # 0,2,3,4 = 9 -> isr is 1
                        10:0 # 1,2,3,4 = 10 -> isr is 0
                        } 
        idx_isr = np.sum(np.stack([comb[f"{i}_m0_c0"],
                                   comb[f"{i}_m0_c1"],
                                   comb[f"{i}_m1_c0"],
                                   comb[f"{i}_m1_c1"]],axis=1),axis=1)
        idx_isr = np.vectorize(where_is_isr.get)(idx_isr)
        
        # verify that the ISR identification is correct
        # for IDX in range(10):
        #     print(comb[f"{i}_m0_c0"][IDX],
        #           comb[f"{i}_m0_c1"][IDX],
        #           comb[f"{i}_m1_c0"][IDX],
        #           comb[f"{i}_m1_c1"][IDX],
        #           idx_isr[IDX])

        # create ISR 4-momenta
        mom4_isr = jets[np.arange(jets.shape[0]),idx_isr]

        # convert to (pt,eta,phi,e)
        mom4_m0_ptetaphi    = PtEtaPhiE(mom4_m0)
        mom4_m1_ptetaphi    = PtEtaPhiE(mom4_m1)
        mom4_m0_c0_ptetaphi = PtEtaPhiE(mom4_m0_c0)
        mom4_m0_c1_ptetaphi = PtEtaPhiE(mom4_m0_c1)
        mom4_m1_c0_ptetaphi = PtEtaPhiE(mom4_m1_c0)
        mom4_m1_c1_ptetaphi = PtEtaPhiE(mom4_m1_c1)
        mom4_isr_ptetaphi   = PtEtaPhiE(mom4_isr)

        # make easy to use dictionaries for parents and children
        m0_c0 = easy_4mom_dict(mom4_m0_c0_ptetaphi)
        m0_c1 = easy_4mom_dict(mom4_m0_c1_ptetaphi)
        m0    = easy_4mom_dict(mom4_m0_ptetaphi)
        m1_c0 = easy_4mom_dict(mom4_m1_c0_ptetaphi)
        m1_c1 = easy_4mom_dict(mom4_m1_c1_ptetaphi)
        m1    = easy_4mom_dict(mom4_m1_ptetaphi)
        isr   = easy_4mom_dict(mom4_isr_ptetaphi)


        # plot the eta distributions
        output[f"{i}_all_final_states_eta"] = np.concatenate([m0_c0["eta"],m0_c1["eta"],m1_c0["eta"],m1_c1["eta"]])
        hist, bin_edges = np.histogram(output[f"{i}_all_final_states_eta"],bins=np.linspace(-10,10, 20 + 1),density=density)
        SaveHist(output,f"{i}_all_final_states_eta",hist,bin_edges)

        # plot the phi distributions
        output[f"{i}_all_final_states_phi"] = np.concatenate([m0_c0["phi"],m0_c1["phi"],m1_c0["phi"],m1_c1["phi"]])
        hist, bin_edges = np.histogram(output[f"{i}_all_final_states_phi"],bins=np.linspace(-4, 4, 16+ 1),density=density)
        SaveHist(output,f"{i}_all_final_states_phi",hist,bin_edges)

        # compute mass asymmetry
        output[f"{i}_m0"] = np.sqrt(mom4_m0[:, 0]**2 - mom4_m0[:, 1]**2 - mom4_m0[:, 2]**2 - mom4_m0[:, 3]**2)
        output[f"{i}_m1"] = np.sqrt(mom4_m1[:, 0]**2 - mom4_m1[:, 1]**2 - mom4_m1[:, 2]**2 - mom4_m1[:, 3]**2)

        output[f"{i}_mass_asymmetry"] = abs(output[f"{i}_m0"] - output[f"{i}_m1"])/(output[f"{i}_m0"] + output[f"{i}_m1"]) 
        bins = np.linspace(0,1,21)
        hist, bin_edges = np.histogram(output[f"{i}_mass_asymmetry"],bins=bins,density=density)
        SaveHist(output,f"{i}_mass_asymmetry",hist,bin_edges)
        output[f"{i}_mass_asymmetry_bin_errs"] = getNp1DBinError(output[f"{i}_mass_asymmetry"],bins,density)

        # compute average mass
        output[f"{i}_mass_avg"] = 0.5*(output[f"{i}_m0"] + output[f"{i}_m1"])
        bins = np.linspace(0, 2*mass, 20 + 1) #np.linspace(mass - 500, mass + 1500, 20 + 1) # Hard coded good bining for 500 GeV
        hist, bin_edges = np.histogram(output[f"{i}_mass_avg"],bins=bins,density=density)
        if fracEvents:
            hist = hist/nevents
        SaveHist(output,f"{i}_mass_avg",hist,bin_edges)
        output[f"{i}_mass_avg_bin_errs"] = getNp1DBinError(output[f"{i}_mass_avg"],bins,density)
        if fracEvents:
            output[f"{i}_mass_avg_bin_errs"] = output[f"{i}_mass_avg_bin_errs"]/nevents

        # compute delta r between pairs
        output[f"{i}_m0_dr"] = np.sqrt((m0_c0["eta"] - m0_c1["eta"])**2 + phys.DeltaPhi(m0_c0["phi"],m0_c1["phi"])**2)
        output[f"{i}_m1_dr"] = np.sqrt((m1_c0["eta"] - m1_c1["eta"])**2 + phys.DeltaPhi(m1_c0["phi"],m1_c1["phi"])**2)
        dr = np.stack([output[f"{i}_m0_dr"],output[f"{i}_m1_dr"]],axis=-1)

        # compute delta r between opposite parents
        output[f"{i}_m0c0_m1c0_dr"] = np.sqrt((m0_c0["eta"] - m1_c0["eta"])**2 + phys.DeltaPhi(m0_c0["phi"],m1_c0["phi"])**2)
        output[f"{i}_m0c0_m1c1_dr"] = np.sqrt((m0_c0["eta"] - m1_c1["eta"])**2 + phys.DeltaPhi(m0_c0["phi"],m1_c1["phi"])**2)
        output[f"{i}_m0c1_m1c0_dr"] = np.sqrt((m0_c1["eta"] - m1_c0["eta"])**2 + phys.DeltaPhi(m0_c1["phi"],m1_c0["phi"])**2)
        output[f"{i}_m0c1_m1c1_dr"] = np.sqrt((m0_c1["eta"] - m1_c1["eta"])**2 + phys.DeltaPhi(m0_c1["phi"],m1_c1["phi"])**2)

        # compute delta r with ISR
        output[f"{i}_m0c0_isr_dr"] = np.sqrt((m0_c0["eta"] - isr["eta"])**2 + phys.DeltaPhi(m0_c0["phi"],isr["phi"])**2)
        output[f"{i}_m0c1_isr_dr"] = np.sqrt((m0_c1["eta"] - isr["eta"])**2 + phys.DeltaPhi(m0_c1["phi"],isr["phi"])**2)
        output[f"{i}_m1c0_isr_dr"] = np.sqrt((m1_c0["eta"] - isr["eta"])**2 + phys.DeltaPhi(m1_c0["phi"],isr["phi"])**2)
        output[f"{i}_m1c1_isr_dr"] = np.sqrt((m1_c1["eta"] - isr["eta"])**2 + phys.DeltaPhi(m1_c1["phi"],isr["phi"])**2)

        # verify the numpy computation of masym and dr versus root
        # check: Passed after update to DeltaPhi method
        nCheckAgainstRoot = 10
        fail = 0
        for iE in range(nCheckAgainstRoot):
            v0 = jets[iE][comb[f"{i}_m0_c0"][iE]]
            v1 = jets[iE][comb[f"{i}_m0_c1"][iE]]
            v0 = ROOT.Math.PxPyPzEVector(v0[1],v0[2],v0[3],v0[0])
            v1 = ROOT.Math.PxPyPzEVector(v1[1],v1[2],v1[3],v1[0])
            dr0 = ROOT.Math.VectorUtil.DeltaR(v0,v1)
            t0 = (v0+v1)
            v2 = jets[iE][comb[f"{i}_m1_c0"][iE]]
            v3 = jets[iE][comb[f"{i}_m1_c1"][iE]]
            v2 = ROOT.Math.PxPyPzEVector(v2[1],v2[2],v2[3],v2[0])
            v3 = ROOT.Math.PxPyPzEVector(v3[1],v3[2],v3[3],v3[0])
            dr1 = ROOT.Math.VectorUtil.DeltaR(v2,v3)
            t1 = (v2+v3)
            rd = 1 # played around with this rd value and it seems the precision of the conversions is roughly 1 decimal place
            if round(output[f"{i}_m0"][iE],rd) != round(t0.M(),rd) or round(output[f"{i}_m1"][iE],rd) != round(t1.M(),rd):
                print("-----------------------------------")
                print("Event %i not equal masses so printing" % (iE))
                print("Numpy computation of mass 0 and 1: (%f,%f)" % (output[f"{i}_m0"][iE],output[f"{i}_m1"][iE]))
                print("Root computation of mass 0 and 1: (%f,%f)" % (t0.M(),t1.M()))
                fail += 1
            rd = 4 # did not vary this value
            if round(dr[iE][0],rd) != round(dr0,rd) or round(dr[iE][1],rd) != round(dr1,rd):
                print("-----------------------------------")
                print("Event %i not equal drs so printing" % (iE))
                print("Numpy computation of dr 0 and 1: (%f,%f)" % (dr[iE][0],dr[iE][1]))
                print("Root computation of dr 0 and 1: (%f,%f)" % (dr0,dr1))
                print("Match (%i,%i) and (%i,%i)" % (comb[f"{i}_m0_c0"][iE], comb[f"{i}_m0_c1"][iE], comb[f"{i}_m1_c0"][iE], comb[f"{i}_m1_c1"][iE]))
                print(jets[iE])
                fail += 1
        if fail:
            print("-----------------------------------")
            print("Number of events with ill-matching numpy and root computation: %i/%i" % (fail,nCheckAgainstRoot))

        # compute dr min for c
        for c in [0.8,1]:
            output[f"{i}_drmin_c{c}"] = np.sum(abs(dr - c),axis=1)
            bins = np.linspace(0,10,21)
            hist, bin_edges = np.histogram(output[f"{i}_drmin_c{c}"],bins=bins,density=density)  
            SaveHist(output,f"{i}_drmin_c{c}",hist,bin_edges)
            output[f"{i}_drmin_c{c}_bin_errs"] = getNp1DBinError(output[f"{i}_drmin_c{c}"],bins,density)

        # compute delta r vs stop pt
        hist, x_edges, y_edges = np.histogram2d(np.concatenate([m0["pt"],m1["pt"]]),
                                                np.concatenate([output[f"{i}_m0_dr"],output[f"{i}_m1_dr"]]),
                                                bins=[np.linspace(0,3000,int(3000/25)+1),
                                                      np.linspace(0,10,101)],
                                                density=density
                                                )  
        SaveHist2D(output,f"{i}_stop_pt_vs_children_dr",hist,x_edges,y_edges)

        # compute delta eta vs delta phi for the same stop parents
        hist, x_edges, y_edges = np.histogram2d(np.concatenate([m0_c0["eta"],m1_c0["eta"]]) - np.concatenate([m0_c1["eta"],m1_c1["eta"]]),
                                                phys.DeltaPhi(np.concatenate([m0_c0["phi"],m1_c0["phi"]]), np.concatenate([m0_c1["phi"],m1_c1["phi"]])),
                                                bins=[np.linspace(-10,10,41),
                                                      np.linspace(-4,4,17)],
                                                density=density
                                                )  
        SaveHist2D(output,f"{i}_deta_vs_dphi_same_parent",hist,x_edges,y_edges)
        # compute delta eta vs delta phi for different stop parents
        hist, x_edges, y_edges = np.histogram2d(np.concatenate([m0_c0["eta"],m0_c0["eta"],m0_c1["eta"],m0_c1["eta"]]) - np.concatenate([m1_c0["eta"],m1_c1["eta"],m1_c0["eta"],m1_c1["eta"]]),
                                                phys.DeltaPhi(np.concatenate([m0_c0["phi"],m0_c0["phi"],m0_c1["phi"],m0_c1["phi"]]), np.concatenate([m1_c0["phi"],m1_c1["phi"],m1_c0["phi"],m1_c1["phi"]])),
                                                bins=[np.linspace(-10,10,41),
                                                      np.linspace(-4,4,17)],
                                                density=density
                                                )  
        SaveHist2D(output,f"{i}_deta_vs_dphi_different_parent",hist,x_edges,y_edges)


        # boost the 4-momenta to center of mass
        mom4_m0_com, mom4_m1_com, boost_com = phys.boost_to_center_of_mass(mom4_m0, mom4_m1)

        # check comparison of numpy vs root calculation
        thresh = 10**-3
        mom4_m0_com_root, mom4_m1_com_root, boost_com_root = phys.boost_to_center_of_mass_root(mom4_m0[:nCheckAgainstRoot], mom4_m1[:nCheckAgainstRoot])
        fail = 0
        for iE in range(nCheckAgainstRoot):
            good = np.sum(mom4_m0_com[iE] - mom4_m0_com_root[iE]) < thresh
            good = good and (np.sum(mom4_m1_com[iE] - mom4_m1_com_root[iE]) < thresh)
            good = good and (np.sum(boost_com[iE] - boost_com_root[iE]) < thresh)
            if not good:
                print("-----------------------------------")
                print("Event %i not equal boosting so printing" % (iE))
                print("Numpy computation of boost to cm")
                print(mom4_m0_com[iE])
                print(mom4_m1_com[iE])
                print(boost_com[iE])
                print("Root computation of boost to cm")
                print(mom4_m0_com_root[iE])
                print(mom4_m1_com_root[iE])
                print(boost_com_root[iE])
                fail += 1
        if fail:
            print("-----------------------------------")
            print("Number of events with ill-matching numpy and root computation: %i/%i" % (fail,nCheckAgainstRoot))

        # convert to (pt,eta,phi,e)
        mom4_m0_com_ptetaphi = PtEtaPhiE(mom4_m0_com)
        mom4_m1_com_ptetaphi = PtEtaPhiE(mom4_m1_com)

        # convert to easy to use dictionaries
        m_com = {"m0_com":easy_4mom_dict(mom4_m0_com_ptetaphi),
                 "m1_com":easy_4mom_dict(mom4_m1_com_ptetaphi)}

        # compute and histogram cos(theta*) for stop 1/2
        for m in [0,1]:
            output[f"{i}_m{m}_com_theta"] = 2*np.arctan2(np.exp(m_com[f"m{m}_com"]["eta"]),-1)
            output[f"{i}_m{m}_com_abs_cos_theta"] = abs(np.cos(output[f"{i}_m{m}_com_theta"]))
            bins = np.linspace(0,1,21)
            hist, bin_edges = np.histogram(output[f"{i}_m{m}_com_abs_cos_theta"],bins=bins,density=density)  
            SaveHist(output,f"{i}_m{m}_com_abs_cos_theta",hist,bin_edges)
            output[f"{i}_m{m}_com_abs_cos_theta_bin_errs"] = getNp1DBinError(output[f"{i}_m{m}_com_abs_cos_theta"],bins,density)

    # save to npz file
    outFileName = os.path.basename(config["comb_file"]).replace(".npz", "_analyze.npz")
    outFileName = os.path.join(config["out_dir"],outFileName)
    np.savez(outFileName, **output)
    return 1

def easy_4mom_dict(mom):
    return {"pt"  : mom[:, 0],
            "eta" : mom[:, 1],
            "phi" : mom[:, 2],
            "e"   : mom[:, 3],}

def PtEtaPhiE(x):
    ''' x is batched [[[e,px,py,pz],[e,px,py,pz],...],...] '''   
    # make into pt eta phi e
    e  = x[:, 0]
    px = x[:, 1]
    py = x[:, 2]
    pz = x[:, 3]
    pt  = np.sqrt(px**2 + py**2)
    eta = np.arcsinh(pz/pt)
    # convert a eta value of nan caused by pt of 0 for padded jets
    # note checking with root computation below the correct value is to send to 0 not inf
    eta = np.nan_to_num(eta,nan=0) 
    phi = np.arctan2(py,px)
    return np.stack([pt,eta,phi,e],axis=-1)

def run_number(file):
    #return os.path.basename(file).split("_")[2]
    return "_".join(file.split("_")[:3])

def SaveHist(output,name,hist,bin_edges):
    output[name + "_hist"]        = hist
    output[name + "_bin_edges"]   = bin_edges
    output[name + "_bin_centers"] = (bin_edges[:-1] + bin_edges[1:]) / 2
    output[name + "_bin_widths"]  = np.diff(bin_edges) 

def SaveHist2D(output,name,hist,x_edges,y_edges):
    output[name + "_hist"]      = hist
    output[name + "_x_edges"]   = x_edges
    output[name + "_x_centers"] = (x_edges[:-1] + x_edges[1:]) / 2
    output[name + "_x_widths"]  = np.diff(x_edges) 
    output[name + "_y_edges"]   = y_edges
    output[name + "_y_centers"] = (y_edges[:-1] + y_edges[1:]) / 2
    output[name + "_y_widths"]  = np.diff(y_edges) 

if __name__ == "__main__":
    main()
