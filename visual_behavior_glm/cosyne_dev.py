import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import visual_behavior_glm.GLM_params as glm_params
plt.ion()

num_vip = [1495,1726,1745,1130]
num_vip_sig = [970,1265,923,763]

num_slc = [12825, 11811, 12668, 11001]
num_slc_sig = [6177,5226,9080,6592]

num_sst = [492,674,534,259]
num_sst_sig = [384,522,389,198]

def plot_coding_fraction(num,sig,w=.45):
    plt.figure(figsize=(6,4))
    frac = np.array(sig)/np.array(num)
    se = 1.98*np.sqrt(frac*(1-frac)/num)
    frac = frac*100
    se = se*100
    for dex, val in enumerate(zip(frac,se)):
        plt.plot([dex-w,dex+w],[val[0],val[0]], 'r',linewidth=4)
        plt.plot([dex,dex],[val[0]+val[1],val[0]-val[1]], 'k',linewidth=1)

    plt.ylabel('% of Vip cells with \n significant coding',fontsize=24)
    plt.xlabel('Session',fontsize=24)
    plt.tick_params(axis='both',labelsize=16)
    plt.xticks([0,1,2,3],['F1','F3','N1','N3'],fontsize=24)
    plt.ylim(30,80)
    plt.tight_layout()

def plot_coding_comparison(w=.45):
    num_vip = [1495,1726,1745,1130]
    num_vip_sig = [970,1265,923,763]
    num_slc = [12825, 11811, 12668, 11001]
    num_slc_sig = [6177,5226,9080,6592]
    num_sst = [492,674,534,259]
    num_sst_sig = [384,522,389,198]

    plt.figure(figsize=(6,4))
    plot_coding_comparison_helper(plt.gca(), num_vip_sig, num_vip,w,'C1')
    plot_coding_comparison_helper(plt.gca(), num_slc_sig, num_slc,w,'C2')
    plot_coding_comparison_helper(plt.gca(), num_sst_sig, num_sst,w,'C0')

    plt.ylabel('% of cells with \n omission coding',fontsize=24)
    plt.xlabel('Session',fontsize=24)
    plt.tick_params(axis='both',labelsize=16)
    plt.xticks([0,1,2,3],['F1','F3','N1','N3'],fontsize=24)
    plt.ylim(30,90)
    plt.tight_layout()

def plot_coding_comparison_helper(ax, sig,num,w,color):
    frac = np.array(sig)/np.array(num)
    se = 1.98*np.sqrt(frac*(1-frac)/num)
    frac = frac*100
    se = se*100
    plt.plot([0,1,2,3], frac,'o-',color=color,linewidth=4)
    for dex, val in enumerate(zip(frac,se)):
        #plt.plot([dex-w,dex+w],[val[0],val[0]], color=color,linewidth=4)
        plt.plot([dex,dex],[val[0]+val[1],val[0]-val[1]], 'k',linewidth=1)



