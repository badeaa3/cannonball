"""
Style a mass asymmetry vs epsilon plot
"""

# general imports
import argparse
import os
import matplotlib.pyplot as plt 
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import ROOT 

# for custom functions
import hnet.plotting.style_combinatoric_solution as scs

# set style based on https://stackoverflow.com/questions/43741928/matplotlib-raw-latex-epsilon-only-yields-varepsilon
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["text.usetex"] = True

def main():

     # argparser
    parser = argparse.ArgumentParser()
    # all model configurations
    parser.add_argument("-c",   help="CoLaLoLa file", default="")
    parser.add_argument("-m",   help="Mass Asymmetry file", default="")
    parser.add_argument("-dr",  help="dR sum file", default="")
    parser.add_argument("-o",   help="Output directory for the resulting files", default="./")
    ops = parser.parse_args()

    # load method comparison data
    files = {"colalola" :ops.c,
             "masym"    :ops.m,
             "drsum1"   :ops.dr}
    
    # load the hist values
    hists = {}
    GeVtoTeV = 1.0/1000
    for key, file in files.items():

        # load hist
        f = ROOT.TFile(file)
        c = f.Get("msu3_vs_ptsmear_test.pdf")
        h = c.GetPrimitive("msu3_vs_ptsmear")

        # get the bin values
        hist = {}
        for ybin in range(h.GetNbinsY()+1)[1:]:
            ykey = round((ybin-1)*0.1,1) # epsilon
            hist[ykey] = {"mass":[],"perf":[]}
            for xbin in range(h.GetNbinsX()+1)[1:]:
                content = h.GetBinContent(xbin,ybin)
                # print (300+xbin*100,(ybin-1)*0.1,content)
                hist[ykey]["mass"].append(round((200+xbin*100)*GeVtoTeV,2))
                hist[ykey]["perf"].append(content)

        # save to dic
        hists[key] = hist
        # print(hist)
        # exit()

    # useful for checking values by hand
    # for key,hist in hists.items():
    #     print(key,list(zip(hist[0.3]["mass"],hist[0.3]["perf"])))

    # draw options
    configs = {"colalola": ["CANNONBALL","#1f77b4",(1.3,0.72),2],
               "masym": [r"$\mathcal{A}$ Minimization", "#ff7f0e",(1.3,0.32),1],
               "drsum1": [r'$\Delta R^{\Sigma}$ (C=1)', "#2ca02c",(1.3,0.07),0]}
    
    # plot
    # fig, ax = plt.subplots(1,1,constrained_layout=False,sharey=False,figsize=(6, 5))
    fig = plt.figure(figsize=(6, 5)) #plt.subplots(1,1,constrained_layout=True,sharey=False,figsize=(6, 5))
    ax = plt.axes([0.1, 0.1, 0.85, 0.85])
    for key,hist in hists.items():

        # plot high, middle, and low epsilon 
        eps = [0.1,0.2,0.3]
        for e in eps:
            ax.plot(hist[e]["mass"], 
                    hist[e]["perf"], 
                    '-', 
                    color=configs[key][1], 
                    alpha=1.0, 
                    zorder=configs[key][3],
                    rasterized=False)

        # fill between high and low epsilon
        ax.fill_between(hist[eps[0]]["mass"], 
                        hist[eps[0]]["perf"], 
                        hist[eps[-1]]["perf"], 
                        alpha=0.2, 
                        color=configs[key][1], 
                        zorder=configs[key][3],
                        rasterized=False)

        # place text above high epsilon
        ax.text(configs[key][2][0],configs[key][2][1],s=configs[key][0],color=configs[key][1],fontsize=14)

    # indication of epsilon boundaries
    mIDX = 5 # index of mass point to put it above
    loc = "masym"
    epsx = hists[loc][eps[1]]["mass"][mIDX]
    epsy = hists[loc][eps[1]]["perf"][mIDX]
    epsyerr = [[hists[loc][eps[1]]["perf"][mIDX] - hists[loc][eps[-1]]["perf"][mIDX]],
              [hists[loc][eps[0]]["perf"][mIDX] - hists[loc][eps[1]]["perf"][mIDX]]]
    
    # ax.arrow(epsx,
    #          hists[loc][eps[0]]["perf"][mIDX] - 0.01,
    #          0,
    #          hists[loc][eps[-1]]["perf"][mIDX] - hists[loc][eps[0]]["perf"][mIDX] + 0.07,
    #          lw=2,
    #          color="black",
    #          alpha=1.0,
    #          head_width = 0.05, 
    #          head_length = 0.05,
    #          overhang=0.2,
    #          rasterized=True)
    ax.errorbar(epsx,epsy,epsyerr,
                fmt="o",
                marker="_",
                markersize=10.5,
                markeredgewidth=2,
                elinewidth=2,
                capsize=5,
                color="black",
                alpha=1.0,
                rasterized=False,
                zorder=100)
    # ax.text(epsx - 0.09, epsy + epsyerr[1][0] + 0.02,  s=r'$\epsilon=$ ' + "%1.1f" % float(eps[0]),  color="black", fontsize=15)
    # ax.text(epsx - 0.06, epsy - epsyerr[0][0] - 0.055,  s="%1.1f" % float(eps[-1]),  color="black", fontsize=15)

    ax.text(epsx - 0.017, epsy + epsyerr[1][0] + 0.02,  s=r'$\epsilon$',  color="black", fontsize=16)
    ax.text(epsx - 0.14, epsy + epsyerr[1][0] - 0.05,  s="%1.1f" % float(eps[0]),  color="black", fontsize=14)
    ax.text(epsx - 0.14, epsy + 0.02,  s="%1.1f" % float(eps[1]),  color="black", fontsize=14)
    ax.text(epsx - 0.14, epsy - epsyerr[0][0] + 0.02,  s="%1.1f" % float(eps[-1]),  color="black", fontsize=14)

    # stamp to be used for labeling the plots
    # stamp  = r'$pp \rightarrow \tilde{t}\tilde{t}(j) \rightarrow$ partons' + ', RPV MG5 aMC@NLO 2.7.3'
    # # stamp += ('\n' + 'MG5 aMC 2.7.3')
    # stamp += ('\n' + r'$\sqrt{s} = 13$ TeV')
    # stamp += ('\n' + r'$p_{\mathrm{T}} \rightarrow \mathcal{N}(p_{\mathrm{T}},$'+epsilon+r'$p_{\mathrm{T}})$')
    ax.text(0.02, 
            1.04,
            r'$pp \rightarrow \tilde{t}\tilde{t}(j) \rightarrow qqqq(j)$' + ', MG5 aMC@NLO',
            horizontalalignment='left',
            verticalalignment='center',
            transform = ax.transAxes,
            fontsize=14,
            linespacing=1.5)


    # set labels
    ax.set_xlabel(r'$\mathrm{m}_{\tilde{t}}$ [TeV]', fontsize=21, labelpad=9)
    ax.set_ylabel('Fraction of Events Correct', fontsize=21, labelpad=9)
    # set limits
    ax.set_xlim(0.3,2.0)
    ax.set_ylim(0,1)
    # set scales
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
    ax.set_xticks(hists["colalola"][eps[1]]["mass"][::3])
    ax.xaxis.set_minor_locator(AutoMinorLocator(3))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(**tick_params_major)
    ax.tick_params(**tick_params_minor)
    # save
    scs.figsave(os.path.join(ops.o, "perfPlot.pdf"))

if __name__ == "__main__":
    main()