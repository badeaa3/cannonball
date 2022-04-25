"""
Style a mass asymmetry vs epsilon plot
"""

# general imports
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt 
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, FuncFormatter)
import ROOT

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
    parser.add_argument("-cqcd",  help="Cannonball QCD Multijet file", default="")
    parser.add_argument("-mqcd",  help="Masym QCD Multijet file", default="")
    parser.add_argument("-drqcd", help="dR sum QCD Multijet file", default="")
    parser.add_argument("-o",   help="Output directory for the resulting files", default="./")
    ops = parser.parse_args()

    # load method comparison data
    files = {"colalola" :ops.c,
             "masym"    :ops.m,
             "drsum1"   :ops.dr,
             "colalola_qcd":ops.cqcd,
             "masym_qcd" : ops.mqcd,
             "drsum1_qcd" : ops.drqcd}
    files = {key:np.load(val,allow_pickle=True) for key,val in files.items() if val != ""}

    # histogram format options
    run = os.path.basename(ops.c).split("_shuf")[0]
    labels = {"truth"    :"Truth",
              "colalola" :"CANNONBALL",
              "masym"    :r'$\mathcal{A}$ Minimization',
              "drsum1"   :r'$\Delta R^{\Sigma}$ (C=1)',
              "colalola_qcd" :None, #"QCD Multijet \nw/CANNONBALL",
              "masym_qcd" :None,#r'w/$\mathcal{A}$ Minimization',
              "drsum1_qcd":None}# r'w/$\Delta R^{\Sigma}$ (C=1)'}

    markers = {"truth"    :"o", # circle
               "colalola" :"D", # diamond
               "masym"    :"s", # square
               "drsum1"   :"X", # x (filled)
               "colalola_qcd" :"D",
               "masym_qcd" : "s",
               "drsum1_qcd" : "X"}

    histcolor = {"truth"    :"black",
                 "colalola" :"#1f77b4", #"#348ABD", #"#1f77b4",
                 "masym"    :"#ff7f0e", #"#E24A33", #"#ff7f0e",
                 "drsum1"   :"#2ca02c", #"#7A68A6", #"#467821", #"#2ca02c",
                 "colalola_qcd" : "#1f77b4", # "#348ABD"
                 "masym_qcd":"#ff7f0e", #"#E24A33", #
                 "drsum1_qcd":"#2ca02c", #"#7A68A6", #"#467821", #
                 }

    '''
    "#E24A33", # orange
    "#7A68A6", # purple
    "#348ABD", # blue
    "#188487", # turquoise
    "#A60628", # red
    "#CF4457", # pink
    "#467821", # green
    '''

    alpha = {"truth"    :1.0,
             "colalola" :1.0,
             "masym"    :1.0,
             "drsum1"   :1.0,
             "colalola_qcd" :0.5,
             "masym_qcd"    :0.2,
             "drsum1_qcd"   :0.2}

    zorder = {"truth"    :0,
              "colalola" :3,
              "masym"    :2,
              "drsum1"   :1,
              "colalola_qcd":-3,
              "masym_qcd"   :-2,
              "drsum1_qcd"  :-1}

    markersize = 9

    # constant conversion factor
    GeVtoTeV = 1.0/1000
    
    # plot
    # fig, ax = plt.subplots(1,1,constrained_layout=False,sharey=False,figsize=(6, 5))
    fig = plt.figure(figsize=(6, 5)) #plt.subplots(1,1,constrained_layout=True,sharey=False,figsize=(6, 5))
    ax = plt.axes([0.1, 0.1, 0.85, 0.85])

    for key,x in files.items():
        print(key)
        # only plot truth once
        if key == "colalola":
            ax.errorbar(x['y_mass_avg_bin_centers'] * GeVtoTeV, 
                        x['y_mass_avg_hist'], 
                        yerr=x['y_mass_avg_bin_errs'],
                        label=labels["truth"],
                        marker=markers["truth"],
                        fmt='^-',
                        markersize=markersize,
                        color=histcolor["truth"],
                        alpha=alpha["truth"],
                        zorder=zorder["truth"])
        # plot errorbars
        if "qcd" not in key:
            ax.errorbar(x['p_mass_avg_bin_centers']  * GeVtoTeV, 
                        x['p_mass_avg_hist'], 
                        yerr=x['p_mass_avg_bin_errs'],
                        label=labels[key],
                        marker=markers[key],
                        fmt='^-',
                        markersize=markersize,
                        color=histcolor[key],
                        alpha=alpha[key],
                        zorder=zorder[key])

        # shade if qcd
        if "qcd" in key:
            if "colalola" in key:
                ax.bar(x['p_mass_avg_bin_centers']  * GeVtoTeV, 
                       x['p_mass_avg_hist'], 
                       align='center', 
                       width=x['p_mass_avg_bin_widths']* GeVtoTeV, 
                       alpha=alpha[key], 
                       color=histcolor[key], 
                       linestyle="-",
                       zorder=zorder[key],
                       label=labels[key],
                       rasterized=True)
            ax.hist(x['p_mass_avg_bin_edges'][:-1]  * GeVtoTeV, 
                    x['p_mass_avg_bin_edges'] * GeVtoTeV,
                    weights=x['p_mass_avg_hist'],
                    alpha=1.0, #0.5, 
                    histtype="step",
                    color=histcolor[key],
                    zorder=zorder[key],
                    rasterized=True)
            #ax.fill_between(x['p_mass_avg_bin_centers']  * GeVtoTeV, x['p_mass_avg_hist'], 0, alpha=0.5, color=histcolor[key], zorder=0)
    
    
    # process stamp
    ax.text(0.02, 
            1.04,
            r'$pp \rightarrow \tilde{t}\tilde{t}(j) \rightarrow qqqq(j)$' + ', MG5 aMC@NLO',
            horizontalalignment='left',
            verticalalignment='center',
            transform = ax.transAxes,
            fontsize=14,
            linespacing=1.5)


    # enable legend
    handles, labels = ax.get_legend_handles_labels()
    # remove error bars 
    handles = [h[0] for h in handles]

    # put qcd at the end
    # handles = handles[1:] + handles[:1]
    # labels = labels[1:] + labels[:1]
    ax.legend(handles,
              labels,
              title=r'$m(\tilde{t})=1$ TeV, $\epsilon=0.2$',
              title_fontsize="12",
              ncol=1,
              framealpha=0.0,
              prop={'size': 10.4},
              bbox_to_anchor=(0.46, 0.45, 0.5, 0.5)) #loc="upper right",
    ax.text(0.12, #0.63, 
            0.895, #0.60,
            "QCD Multijet \n" + r'$pp \rightarrow jjjj(j)$',
            # r"Dimmed $pp \rightarrow jjjj(j)$" + "\n (QCD Multijet)",
            horizontalalignment='left',
            verticalalignment='center',
            transform = ax.transAxes,
            fontsize=12,
            linespacing=1.5,
            alpha = 1.0)
    # set labels
    ax.set_xlabel(r'$\mathrm{m}_{\mathrm{avg}}$ [TeV]', fontsize=21, labelpad=9) # = (m_{1}+m_{2})/2
    ax.set_ylabel("Normalized Event Density", fontsize=21, labelpad=9) # Normalized Event Fraction "Fraction of Events / 100 GeV"
    
    # set limits
    ax.set_xlim(0,2.1)
    ax.set_ylim(10**-5,9.5*10**-3)
    
    # set scales
    ax.set_yscale('log')
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
    tickFormatter = lambda x,pos: str(x).rstrip('0').rstrip('.')
    ax.xaxis.set_major_formatter(FuncFormatter(tickFormatter))
    # plt.yticks(rotation=45, ha='right')
    # save
    plt.savefig(os.path.join(ops.o, "perfmavg.pdf"), bbox_inches="tight")

if __name__ == "__main__":
    main()