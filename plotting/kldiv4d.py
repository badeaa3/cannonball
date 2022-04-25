# general imports
import argparse
import os
import matplotlib.pyplot as plt 
from matplotlib.ticker import (AutoMinorLocator, FuncFormatter)
import numpy as np 
import scipy.stats
import scipy.special

# set style based on https://stackoverflow.com/questions/43741928/matplotlib-raw-latex-epsilon-only-yields-varepsilon
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["text.usetex"] = True

def main():

     # argparser
     parser = argparse.ArgumentParser()
     # all model configurations
     parser.add_argument("-d",   help="Data file", default="")
     parser.add_argument("-c",   help="CoLaLoLa file", default="")
     parser.add_argument("-m",   help="Mass Asymmetry file", default="")
     parser.add_argument("-dr",  help="dR sum file", default="")
     parser.add_argument("-o",   help="Output directory for the resulting files", default="./")
     ops = parser.parse_args()

     # load data
     jets = np.load(ops.d,allow_pickle=True)["jets"]

     # load method comparison data
     files = {"colalola" :ops.c,
              "masym"    :ops.m,
              "drsum1"   :ops.dr}
     files = {key:np.load(val,allow_pickle=True)["combinatoric_solution"].item() for key,val in files.items() if val}

     # pt,eta,phi binning for 3d-histogram
     ptbins = np.linspace(0,2000,11)
     etabins = np.linspace(-5,5,21)
     phibins = np.linspace(-3.5,3.5,21)
     mbins = np.linspace(0,2000,21)
     
     # process combinatoric solutions
     hists = {}     
     for method,comb in files.items():
          
          # convert to numpy
          for key,val in comb.items():
               if not isinstance(val,np.ndarray):
                    comb[key] = np.array(val)  

          # prepare dictionary
          hists[method] = {}

          # loop over prediction and truth
          for iPY in ["p","y"]:

               # extract the children 4-momenta in (px,py,pz,e)
               t0_c0 = jets[np.arange(jets.shape[0]),comb[f"{iPY}_m0_c0"].astype(int)]
               t0_c1 = jets[np.arange(jets.shape[0]),comb[f"{iPY}_m0_c1"].astype(int)]
               t1_c0 = jets[np.arange(jets.shape[0]),comb[f"{iPY}_m1_c0"].astype(int)]
               t1_c1 = jets[np.arange(jets.shape[0]),comb[f"{iPY}_m1_c1"].astype(int)]

               # create parent 4-momenta
               t0 = t0_c0 + t0_c1
               t1 = t1_c0 + t1_c1
               t = np.stack([t0,t1],axis=1)

               # convert to (pt,eta,phi,e)
               t = PtEtaPhiM(t)

               # make 3d histogram
               # H, edges = np.histogramdd(t[:,:,:3].reshape(t.shape[0]*t.shape[1],3), bins = (ptbins, etabins, phibins), density=False)
               H, edges = np.histogramdd(t[:,:,[0,1,2,3]].reshape(t.shape[0]*t.shape[1],4), bins = (ptbins, etabins, phibins, mbins), density=False)
               # normalize by number of stops to make probability distribution
               H = H/np.sum(H)
               
               # set zero bin to very small number so no infinity in division
               H[H==0] = 10**-20

               # fill dictionary
               hists[method][iPY] = {"H":H, "edges":edges}

     # compute kl_div
     kldiv = {}
     for method, mval in hists.items():
          
          total = scipy.stats.entropy(mval["y"]["H"].flatten(),mval["p"]["H"].flatten())
          print("%s total KL div: %1.4f" % (method,total))
          
          # prepare 
          kldiv[method] = {"total":total,
                           "pt":[], "eta":[], "phi":[]}
          kldiv["truth"] = {"total":scipy.stats.entropy(mval["y"]["H"].flatten(),mval["y"]["H"].flatten()),
                            "pt":[], "eta":[], "phi":[]}

          # loop over pt bins
          for iPt in range(len(ptbins)-1):
               kldiv[method]["pt"].append(np.sum(scipy.special.rel_entr(mval["y"]["H"][iPt].flatten(),mval["p"]["H"][iPt].flatten())))
               kldiv["truth"]["pt"].append(np.sum(scipy.special.rel_entr(mval["y"]["H"][iPt].flatten(),mval["y"]["H"][iPt].flatten())))

          # loop over eta bins
          for iEta in range(len(etabins)-1):
               kldiv[method]["eta"].append(np.sum(scipy.special.rel_entr(mval["y"]["H"][:,iEta].flatten(),mval["p"]["H"][:,iEta].flatten())))
               kldiv["truth"]["eta"].append(np.sum(scipy.special.rel_entr(mval["y"]["H"][:,iEta].flatten(),mval["y"]["H"][:,iEta].flatten())))

          # loop over phi bins
          for iPhi in range(len(phibins)-1):
               kldiv[method]["phi"].append(np.sum(scipy.special.rel_entr(mval["y"]["H"][:,:,iPhi].flatten(),mval["p"]["H"][:,:,iPhi].flatten())))
               kldiv["truth"]["phi"].append(np.sum(scipy.special.rel_entr(mval["y"]["H"][:,:,iPhi].flatten(),mval["y"]["H"][:,:,iPhi].flatten())))

          # print("Total KL div (pt,eta,phi): (%1.4f, %1.4f, %1.4f)" % (np.sum(kldiv[method]["pt"]), np.sum(kldiv[method]["eta"]), np.sum(kldiv[method]["phi"])))

     # plotting configs
     configs = {"truth":[r"$\mathrm{Truth}$","o","black",0],
                "colalola": ["CANNONBALL","D","#1f77b4",3],
                "masym": [r"$\mathcal{A}\; \mathrm{Minimization}$","s","#ff7f0e",2],
                "drsum1": [r"$\Delta R^{\Sigma}\; \mathrm{\left(C=1\right)}$","X", "#2ca02c",1]}
     # constant conversion factor
     GeVtoTeV = 1.0/1000
     # bin centers
     centers = {"pt": (ptbins[:-1] + ptbins[1:]) / 2,
                "eta": (etabins[:-1] + etabins[1:]) / 2,
                "phi": (phibins[:-1] + phibins[1:]) / 2,
                "m": (mbins[:-1] + mbins[1:]) / 2}

     # 2d histograms
     # fig = plt.figure()
     # ax = plt.axes(projection='3d')
     # X, Y = np.meshgrid(centers["eta"], centers["phi"])
     # ax.plot_surface(X, Y, hists["drsum1"]["p"]["H"][3], rstride=1, cstride=1, cmap='jet', edgecolor='none')
     # ax.view_init(60, 35)
     # ax.set_axis_off()
     # # ax.zaxis.set_visible(False) #set_pane_color((1.0, 1.0, 1.0, 0.0))
     # plt.show()

     # exit()

     # plot pt 
     fig = plt.figure(figsize=(6, 5)) #plt.subplots(1,1,constrained_layout=True,sharey=False,figsize=(6, 5))
     ax = plt.axes([0.1, 0.1, 0.85, 0.85])
     # plot in desired order
     for method in ["truth","colalola","masym","drsum1"]: #,kld in kldiv.items():       
          total = "\n" + r"$\left(%1.2f\right)$" % kldiv[method]["total"]
          if method == "truth":
               total = "\n" + r"$\left(\mathcal{D}_{\mathrm{KL}}={%1.2f}\right)$" % kldiv[method]["total"]
          ax.errorbar(centers["pt"] * GeVtoTeV, 
                      kldiv[method]["pt"], 
                      yerr=0,
                      label=configs[method][0] + total,
                      marker=configs[method][1],
                      markersize=9,
                      color=configs[method][2],
                      alpha=1,
                      fmt='^-',
                      zorder=configs[method][3])
     # process stamp
     ax.text(0.02, #0.345
             1.04,
             r'$pp \rightarrow \tilde{t}\tilde{t}(j) \rightarrow qqqq(j)$' + ", MG5 aMC@NLO",
             horizontalalignment='left',
             verticalalignment='center',
             transform = ax.transAxes,
             fontsize=14,
             linespacing=1.5)
     
     # enable legend
     handles, labels = ax.get_legend_handles_labels()
     # put qcd at the end
     # handles = [handles[0],handles[1],handles[-1],handles[2]]
     # labels = [labels[0],labels[1],labels[-1],labels[2]]
     # remove error bars 
     handles = [h[0] for h in handles]
     ax.legend(handles,
               labels,
               title=r'$m(\tilde{t})=1$ TeV, $\epsilon=0.2$',
               title_fontsize="15",
               ncol=1,
               framealpha=0.0,
               prop={'size': 12},
               bbox_to_anchor=(0.44, 0.4, 0.5, 0.5),
               labelspacing=1,
               columnspacing=1) #loc="upper right",
     # ax.text(0.5,0.91,r'$m(\tilde{t})=1$ TeV, $\epsilon=0.2$',fontsize=11,transform = ax.transAxes))
     # set labels
     ax.set_xlabel(r'$p_{\mathrm{T}}^{\tilde{t}} \; \mathrm{[TeV]}$', fontsize=21, labelpad=3)
     ax.set_ylabel(r'$\mathcal{D}_{\mathrm{KL}}^{\eta,\phi,\mathrm{m}}\left(T\;||\;P, p_{\mathrm{T}}\right)$', fontsize=21, labelpad=9) # r'$\mathcal{D}_{\mathrm{KL}}$ (Truth || Pred.) in $(\eta,\phi, \mathrm{m})$'
     # set limits
     ax.set_xlim(0,2.05)
     ax.set_ylim(-0.01,0.18)
     # set scale
     ax.set_yscale('linear')
     # enable grid
     # ax.grid()
     # style ticks
     tick_params_major = {"which"     : "major",
                         "length"    : 10,
                         "width"     : 1.25,
                         "direction" : "in",
                         "right"     : True,
                         "top"       : True,
                         "labelsize" : 15,
                         "pad"       : 8}
     tick_params_minor = {"which"     : "minor",
                         "length"    : 6,
                         "width"     : 1.2,
                         "direction" : "in",
                         "right"     : True,
                         "top"       : True,
                         "labelsize" : 15}
     ax.set_yticks([0,0.1])
     ax.set_xticks(np.round(np.linspace(0,2,6),2))
     ax.xaxis.set_minor_locator(AutoMinorLocator(2))
     ax.yaxis.set_minor_locator(AutoMinorLocator(4))
     ax.tick_params(**tick_params_major)
     ax.tick_params(**tick_params_minor)
     tickFormatter = lambda x,pos: str(x).rstrip('0').rstrip('.')
     ax.xaxis.set_major_formatter(FuncFormatter(tickFormatter))

     
     # draw 3d hists
     # ax3d = []
     # X, Y = np.meshgrid(centers["eta"], centers["m"])
     # topleft = np.array([0.43, 0.63, 0.2, 0.2])
     # move = np.array([0.22,0.23,0,0]) #0.3,0.27
     # step = {"truth":(0,0,0,0),
     #         "colalola":(1,0,0,0),
     #         "masym":(1,-1,0,0),
     #         "drsum1":(0,-1,0,0)}
     # hists["truth"] = {"p":hists["colalola"]["y"]}
     # for method in ["truth","colalola","masym","drsum1"]:
     #      pos = topleft + step[method] * move
     #      a = plt.axes(pos,projection="3d")
     #      i=3
     #      a.plot_surface(X[i:-i], Y[i:-i], hists[method]["p"]["H"][3][i:-i], cmap='coolwarm')
     #      # a.imshow(hists[method]["p"]["H"][3],cmap="jet")
     #      a.view_init(35, 25)
     #      a.patch.set_alpha(0.0)
     #      # a.set_axis_off()
     #      a.xaxis.set_ticklabels([])
     #      a.yaxis.set_ticklabels([])
     #      a.zaxis.set_ticklabels([])
          
     #      # make transparent
     #      a.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
     #      a.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
     #      a.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
     #      # make the grid lines transparent
     #      a.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
     #      a.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
     #      a.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
     #      # ax.text(pos[0] + 0.05,pos[1] - 0.13, r'$0.3 \leq p_{\mathrm{T}} \leq 0.4$ TeV',fontsize=8,transform = ax.transAxes)
     #      if method == "truth":
     #           a.set_xlabel(r'$\phi$',fontsize=8, labelpad=-15)
     #           a.set_ylabel(r'$\eta$',fontsize=8, labelpad=-15)
     #           a.zaxis.set_rotate_label(False)
     #           a.set_zlabel(r'N$_{\tilde{t}}$',fontsize=8, labelpad=-15, rotation=0)
     #      ax3d.append(a)

     # save
     plt.savefig(os.path.join(ops.o, "kld_pt_etaphim.pdf"), bbox_inches="tight")

     # plot eta 
     fig, ax = plt.subplots(1,1,constrained_layout=True,sharey=False,figsize=(6, 5))
     for method,kld in kldiv.items():           
          ax.errorbar(centers["eta"], 
                      kld["eta"], 
                      yerr=0,
                      label=configs[method][0],
                      marker=configs[method][1],
                      markersize=9,
                      color=configs[method][2],
                      alpha=1,
                      fmt='^-')
     ax.legend(title=r'$m(\tilde{t})=1$ TeV, $\epsilon=0.2$',
               title_fontsize="12",
               ncol=1,
               framealpha=0.0,
               prop={'size': 10.4},
               bbox_to_anchor=(0.48, 0.45, 0.5, 0.5)) #loc="upper right",
     # set labels
     ax.set_xlabel(r'$\eta^{\tilde{t}}$', fontsize=18, labelpad=9)
     ax.set_ylabel(r'$\mathcal{D}_{\mathrm{KL}}$ (Truth || Model) in $(p_{\mathrm{T}},\phi)$', fontsize=18, labelpad=9)
     # set limits
     # ax.set_xlim(0,2)
     # ax.set_ylim(-0.075,0.45)
     # set scale
     ax.set_yscale('linear')
     # enable grid
     # ax.grid()
     # style ticks
     tick_params_major = {"which"     : "major",
                         "length"    : 10,
                         "width"     : 1.25,
                         "direction" : "in",
                         "right"     : True,
                         "top"       : True,
                         "labelsize" : 15,
                         "pad"       : 8}
     tick_params_minor = {"which"     : "minor",
                         "length"    : 4,
                         "width"     : 1,
                         "direction" : "in",
                         "right"     : True,
                         "top"       : True,
                         "labelsize" : 15}
     # ax.set_yticks([0,0.1,0.2,0.3,0.4])
     ax.xaxis.set_minor_locator(AutoMinorLocator(5))
     # ax.yaxis.set_minor_locator(AutoMinorLocator(5))
     ax.tick_params(**tick_params_major)
     ax.tick_params(**tick_params_minor)
     # save
     plt.savefig(os.path.join(ops.o, "kld_eta2.pdf"), bbox_inches="tight")

     # plot phi 
     fig, ax = plt.subplots(1,1,constrained_layout=True,sharey=False,figsize=(6, 5))
     for method,kld in kldiv.items():           
          ax.errorbar(centers["phi"], 
                      kld["phi"], 
                      yerr=0,
                      label=configs[method][0],
                      marker=configs[method][1],
                      markersize=9,
                      color=configs[method][2],
                      alpha=1,
                      fmt='^-')
     ax.legend(title=r'$m(\tilde{t})=1$ TeV, $\epsilon=0.2$',
               title_fontsize="12",
               ncol=1,
               framealpha=0.0,
               prop={'size': 10.4},
               bbox_to_anchor=(0.48, 0.45, 0.5, 0.5)) #loc="upper right",
     # set labels
     ax.set_xlabel(r'$\phi^{\tilde{t}}$', fontsize=18, labelpad=9)
     ax.set_ylabel(r'$\mathcal{D}_{\mathrm{KL}}$ (Truth || Model) in $(p_{\mathrm{T}},\eta)$', fontsize=18, labelpad=9)
     # set limits
     # ax.set_xlim(0,2)
     # ax.set_ylim(-0.01,0.01)
     # set scale
     ax.set_yscale('linear')
     # enable grid
     # ax.grid()
     # style ticks
     tick_params_major = {"which"     : "major",
                         "length"    : 10,
                         "width"     : 1.25,
                         "direction" : "in",
                         "right"     : True,
                         "top"       : True,
                         "labelsize" : 15,
                         "pad"       : 8}
     tick_params_minor = {"which"     : "minor",
                         "length"    : 4,
                         "width"     : 1,
                         "direction" : "in",
                         "right"     : True,
                         "top"       : True,
                         "labelsize" : 15}
     # ax.set_yticks([0,0.1,0.2,0.3,0.4])
     ax.xaxis.set_minor_locator(AutoMinorLocator(5))
     # ax.yaxis.set_minor_locator(AutoMinorLocator(5))
     ax.tick_params(**tick_params_major)
     ax.tick_params(**tick_params_minor)
     # save
     plt.savefig(os.path.join(ops.o, "kld_phi2.pdf"), bbox_inches="tight") 

def PtEtaPhiE(x):
    ''' x is batched [[[e,px,py,pz],[e,px,py,pz],...],...] '''   
    # make into pt eta phi e
    e  = x[:, :, 0]
    px = x[:, :, 1]
    py = x[:, :, 2]
    pz = x[:, :, 3]
    pt  = np.sqrt(px**2 + py**2)
    eta = np.arcsinh(pz/pt)
    # convert a eta value of nan caused by pt of 0 for padded jets
    # note checking with root computation below the correct value is to send to 0 not inf
    eta = np.nan_to_num(eta,nan=0) 
    phi = np.arctan2(py,px)
    return np.stack([pt,eta,phi,e],axis=-1)

def PtEtaPhiM(x):
    ''' x is batched [[[e,px,py,pz],[e,px,py,pz],...],...] '''   
    # make into pt eta phi e
    e  = x[:, :, 0]
    px = x[:, :, 1]
    py = x[:, :, 2]
    pz = x[:, :, 3]
    pt  = np.sqrt(px**2 + py**2)
    eta = np.arcsinh(pz/pt)
    # convert a eta value of nan caused by pt of 0 for padded jets
    # note checking with root computation below the correct value is to send to 0 not inf
    eta = np.nan_to_num(eta,nan=0) 
    phi = np.arctan2(py,px)
    m = np.sqrt(e**2 - px**2 - py**2 - pz**2)
    return np.stack([pt,eta,phi,m],axis=-1)

if __name__ == "__main__":
     main()

     
