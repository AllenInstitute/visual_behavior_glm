import os
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib
from matplotlib import cm
from matplotlib.collections import LineCollection
import visual_behavior_glm_strategy.GLM_dataset as gd
import visual_behavior_glm_strategy.GLM_strategy_tools as gst

CRE = ['Slc17a7-IRES2-Cre','Sst-IRES-Cre','Vip-IRES-Cre']
GLM_VERSION = '12_dff_L2_optimize_by_session'
# TODO
# direction arrows on traces
# need to include error bars on traces
# dataset code is horrible redundant
# dataset code is only looking at layer 2/3 in V1. Should make that more flexible
# make easy to save out comparisons

def get_all(other=None):
    data_f = get_dataset(image_set='familiar',normalize='none',other=other)
    data_fp = get_dataset(image_set='fpassive',normalize='none',other=other)
    data_n = get_dataset(image_set='novel',normalize='none',other=other)
    data_p = get_dataset(image_set='novelp',normalize='none',other=other)
    data_pp = get_dataset(image_set='nppassive',normalize='none',other=other)
    return data_f, data_fp,data_n, data_p,data_pp

def get_strategy():
    data_f_visual = get_dataset(image_set='familiar',normalize='none',other='visual')
    data_n_visual = get_dataset(image_set='novel',normalize='none',other='visual')
    data_p_visual = get_dataset(image_set='novelp',normalize='none',other='visual')
    data_f_timing = get_dataset(image_set='familiar',normalize='none',other='timing')
    data_n_timing = get_dataset(image_set='novel',normalize='none',other='timing')
    data_p_timing = get_dataset(image_set='novelp',normalize='none',other='timing')
    return data_f_visual, data_n_visual, data_p_visual, data_f_timing, data_n_timing,data_p_timing

def save_strategy_data_summary(run_params, weights_df):
    weights_beh = gst.add_behavior_metrics(weights_df.copy())
    weights_beh_visual = weights_beh.query('(strategy == "visual")&(strategy_matched)').copy()
    weights_beh_timing = weights_beh.query('(strategy == "timing")&(strategy_matched)').copy()
    gd.save_data_summary(run_params, weights_beh_visual, normalize='none',other='visual')
    gd.save_data_summary(run_params, weights_beh_timing, normalize='none',other='timing')

def get_dataset(image_set='familiar',other=None,normalize='none'):

    dataset = gd.make_dataset(image_set=image_set,other=other,normalize=normalize)


    if image_set in ['familiar','novel','novelp']:
        trim = ['composite_images','composite_omission','composite_change']
    else:
        trim = ['composite_images','composite_omission']
    for t in trim:
        for k in dataset[t]:
            if k != 'meta':
                dataset[t][k] = dataset[t][k][96:]
            if k == 'time':
                dataset[t][k] = dataset[t][k]-3
    return dataset

def compare_regressed(data_f, data_fp, data_n, data_p, data_pp):
    ax = plot_3D(data_f, 'omissions', color='r',label='Familiar')
    ax = plot_3D(data_n, 'omissions', color='b',label='Novel',ax=ax)
    ax = plot_3D(data_p, 'omissions', color='m',label='Novel +1',ax=ax)
    plt.legend()
    save_figure(plt.gcf(),GLM_VERSION, 'compare_omissions_regression_3d') 

    ax = plot_proj(data_f, 'omissions')
    ax = plot_proj(data_n, 'omissions', cmap='winter_r',axes=ax)
    ax = plot_proj(data_p, 'omissions', cmap='cool',axes=ax)
    plt.legend()
    save_figure(plt.gcf(),GLM_VERSION, 'compare_omissions_regression_proj') 


def compare_passive(data_f, data_fp, data_n, data_p, data_pp):
    ax = plot_3D(data_f, 'composite_images', color='r', label='Familiar')
    ax = plot_3D(data_fp, 'composite_images', color='tab:pink', label='Familiar Passive',ax=ax)
    plt.legend()
    save_figure(plt.gcf(),GLM_VERSION, 'familiar_passive_image_3d') 

    ax = plot_3D(data_p, 'composite_images', color='m', label='Novel +1')
    ax = plot_3D(data_pp, 'composite_images', color='c', label='Novel Passive',ax=ax)
    plt.legend()
    save_figure(plt.gcf(),GLM_VERSION, 'novel_passive_image_3d') 

    ax = plot_3D(data_f, 'composite_images', color='r', label='Familiar')
    ax = plot_3D(data_fp, 'composite_images', color='tab:pink', label='Familiar Passive',ax=ax)
    ax = plot_3D(data_p, 'composite_images', color='m', label='Novel +1',ax=ax)
    ax = plot_3D(data_pp, 'composite_images', color='c', label='Novel Passive',ax=ax)
    plt.legend()
    save_figure(plt.gcf(),GLM_VERSION, 'passive_image_3d') 

    ax = plot_proj(data_f, 'composite_images',arrow=True)
    ax = plot_proj(data_fp,'composite_images',arrow=True,cmap='pink_r',axes=ax)
    save_figure(plt.gcf(),GLM_VERSION, 'familiar_passive_image_proj') 

    ax = plot_proj(data_p, 'composite_images',arrow=True,cmap='pink_r')
    ax = plot_proj(data_pp, 'composite_images',arrow=True,cmap='winter_r',axes=ax)
    save_figure(plt.gcf(),GLM_VERSION, 'novel_passive_image_proj') 

    ax = plot_proj(data_f, 'composite_images',arrow=True)
    ax = plot_proj(data_fp,'composite_images',arrow=True,cmap='pink_r',axes=ax)
    ax = plot_proj(data_p, 'composite_images',arrow=True,cmap='pink_r',axes=ax)
    ax = plot_proj(data_pp, 'composite_images',arrow=True,cmap='winter_r',axes=ax)
    save_figure(plt.gcf(),GLM_VERSION, 'passive_image_proj') 


def compare_strategy(dataset_f_visual, dataset_f_timing):
    ax = plot_3D(dataset_f_visual, 'composite_images',color='tab:orange',label='Visual')
    ax = plot_3D(dataset_f_timing, 'composite_images',color='tab:purple',ax=ax,label='Timing')
    plt.legend()
    save_figure(plt.gcf(),GLM_VERSION, 'familiar_strategy_image_3d') 

    ax = plot_3D(dataset_f_visual, 'composite_omission',color='tab:orange',label='Visual')
    ax = plot_3D(dataset_f_timing, 'composite_omission',color='tab:purple',ax=ax,label='Timing')
    plt.legend()
    save_figure(plt.gcf(),GLM_VERSION, 'familiar_strategy_omission_3d') 

    ax = plot_3D(dataset_f_visual, 'composite_change',color='tab:orange',label='Visual')
    ax = plot_3D(dataset_f_timing, 'composite_change',color='tab:purple',ax=ax,label='Timing')
    plt.legend()
    save_figure(plt.gcf(),GLM_VERSION, 'familiar_strategy_change_3d') 

def compare_against(dataset_f,dataset_n):
    ax = plot_3D(dataset_n, 'composite_omission', color='b')
    ax = plot_3D(dataset_n, 'composite_images',ax=ax,color='m')

    ax = plot_proj(dataset_n, 'composite_omission', cmap='winter_r',arrow=False)
    ax = plot_proj(dataset_n, 'composite_images',axes=ax,arrow=True,cmap='cool')

    ax = plot_3D(dataset_f, 'composite_omission', color='r')
    ax = plot_3D(dataset_f, 'composite_images',ax=ax,color='m')

    ax = plot_proj(dataset_f, 'composite_omission', arrow=False)
    ax = plot_proj(dataset_f, 'composite_images',axes=ax,arrow=True,cmap='cool')

def compare(dataset_f, dataset_n):
    ax = plot_proj(dataset_f, 'composite_images',arrow=True)
    ax = plot_proj(dataset_n, 'composite_images',arrow=True,axes=ax,cmap='winter_r')

def compare_images(dataset_f, dataset_n,dataset_p,images=True,omissions=True, changes=True):
    # Images
    if images:
        ax = plot_3D(dataset_p, 'composite_images',color='m')
        save_figure(plt.gcf(),GLM_VERSION, 'novelp_image_3d') 
        ax = plot_3D(dataset_n, 'composite_images',color='b')
        save_figure(plt.gcf(),GLM_VERSION, 'novel_image_3d') 
        ax = plot_3D(dataset_f, 'composite_images')
        save_figure(plt.gcf(),GLM_VERSION, 'familiar_image_3d') 
        ax = plot_3D(dataset_n, 'composite_images',ax=ax, color='b')
        save_figure(plt.gcf(),GLM_VERSION, 'compare_image_3d')
        ax = plot_3D(dataset_p, 'composite_images',ax=ax, color='m')
        save_figure(plt.gcf(),GLM_VERSION, 'compare3_image_3d') 
    
        # Comparing familiar/novel images
        ax = plot_proj(dataset_p, 'composite_images',arrow=True,cmap='cool')
        save_figure(plt.gcf(),GLM_VERSION, 'novelp_image_proj') 
        ax = plot_proj(dataset_n, 'composite_images',arrow=True,cmap='winter_r')
        save_figure(plt.gcf(),GLM_VERSION, 'novel_image_proj') 
        ax = plot_proj(dataset_f, 'composite_images',arrow=True)
        save_figure(plt.gcf(),GLM_VERSION, 'familiar_image_proj') 
        ax = plot_proj(dataset_n, 'composite_images',axes=ax, cmap='winter_r',arrow=True)
        save_figure(plt.gcf(),GLM_VERSION, 'compare_image_proj') 
        ax = plot_proj(dataset_p, 'composite_images',axes=ax, cmap='cool',arrow=True)
        save_figure(plt.gcf(),GLM_VERSION, 'compare3_image_proj') 


    #omissions
    if omissions:
        ax = plot_3D(dataset_p, 'composite_omission', color='m')
        save_figure(plt.gcf(),GLM_VERSION, 'novelp_omission_3d') 
        ax = plot_3D(dataset_n, 'composite_omission', color='b')
        save_figure(plt.gcf(),GLM_VERSION, 'novel_omission_3d') 
        ax = plot_3D(dataset_f, 'composite_omission')
        save_figure(plt.gcf(),GLM_VERSION, 'familiar_omission_3d') 
        ax = plot_3D(dataset_n, 'composite_omission',ax=ax, color='b')
        save_figure(plt.gcf(),GLM_VERSION, 'compare_omission_3d') 
        ax = plot_3D(dataset_p, 'composite_omission',ax=ax, color='m')
        save_figure(plt.gcf(),GLM_VERSION, 'compare3_omission_3d') 
    
        # Comparing familar/novel omissions
        ax = plot_proj(dataset_p, 'composite_omission', cmap='cool',arrow=False)
        save_figure(plt.gcf(),GLM_VERSION, 'novelp_omission_proj') 
        ax = plot_proj(dataset_n, 'composite_omission', cmap='winter_r',arrow=False)
        save_figure(plt.gcf(),GLM_VERSION, 'novel_omission_proj') 
        ax = plot_proj(dataset_f, 'composite_omission',arrow=False)
        save_figure(plt.gcf(),GLM_VERSION, 'familiar_omission_proj') 
        ax = plot_proj(dataset_n, 'composite_omission',axes=ax, cmap='winter_r',arrow=False)
        save_figure(plt.gcf(),GLM_VERSION, 'compare_omission_proj') 
        ax = plot_proj(dataset_p, 'composite_omission',axes=ax, cmap='cool',arrow=False)
        save_figure(plt.gcf(),GLM_VERSION, 'compare3_omission_proj') 

    # Change
    if changes:
        ax = plot_3D(dataset_p, 'composite_change', color='m')
        save_figure(plt.gcf(),GLM_VERSION, 'novelp_change_3d')
        ax = plot_3D(dataset_n, 'composite_change', color='b')
        save_figure(plt.gcf(),GLM_VERSION, 'novel_change_3d') 
        ax = plot_3D(dataset_f, 'composite_change')
        save_figure(plt.gcf(),GLM_VERSION, 'familiar_change_3d') 
        ax = plot_3D(dataset_n, 'composite_change',ax=ax, color='b')
        save_figure(plt.gcf(),GLM_VERSION, 'compare_change_3d') 
        ax = plot_3D(dataset_p, 'composite_change',ax=ax, color='m')
        save_figure(plt.gcf(),GLM_VERSION, 'compare3_change_3d') 
    
        # Comparing familiar/novel change
        ax = plot_proj(dataset_p, 'composite_change', cmap='cool',arrow=False)
        save_figure(plt.gcf(),GLM_VERSION, 'novelp_change_proj') 
        ax = plot_proj(dataset_n, 'composite_change', cmap='winter_r',arrow=False)
        save_figure(plt.gcf(),GLM_VERSION, 'novel_change_proj') 
        ax = plot_proj(dataset_f, 'composite_change',arrow=False)
        save_figure(plt.gcf(),GLM_VERSION, 'familiar_change_proj') 
        ax = plot_proj(dataset_n, 'composite_change',axes=ax, cmap='winter_r',arrow=False)
        save_figure(plt.gcf(),GLM_VERSION, 'compare_change_proj') 
        ax = plot_proj(dataset_p, 'composite_change',axes=ax, cmap='cool',arrow=False)
        save_figure(plt.gcf(),GLM_VERSION, 'compare3_change_proj') 

def plot_3D(dataset, kernels,ax=None,color='r',alpha=1,label=None):
    add_box=False
    if ax is None:
        add_box=False
        fig = plt.figure()
        ax = plt.axes(projection='3d')
    line = ax.plot3D(dataset[kernels][CRE[0]],dataset[kernels][CRE[1]],dataset[kernels][CRE[2]],linewidth=2,color=color,alpha=alpha,label=label)
    if 'composite' in kernels:
        im = np.mod(dataset[kernels]['time'],0.75) == 0
    else:
        im = np.mod(dataset[kernels]['time'],0.75) < 0.03
    ax.scatter(dataset[kernels][CRE[0]][im],dataset[kernels][CRE[1]][im],dataset[kernels][CRE[2]][im],c=dataset[kernels]['time'][im],cmap='autumn_r',edgecolor='k',alpha=1)
    ax.set_xlabel(CRE[0])
    ax.set_ylabel(CRE[1])
    ax.set_zlabel(CRE[2])
    if 'omission' in kernels:
        ax.view_init(elev=25, azim=-45)
        if add_box:
            r = np.max(np.abs(dataset[kernels][CRE[0]]))
            xx,yy = np.meshgrid(np.linspace(-1,1.5), np.linspace(-1,8))
            ax.plot_surface(xx,xx,yy,alpha=.25)
    else:
        #ax.set_xlim(-1,1)    
        #ax.set_ylim(-1,1)
        #ax.set_zlim(-1,1) 
        ax.view_init(elev=35, azim=-45)
        if add_box:
            r = np.max(np.abs(dataset[kernels][CRE[0]]))
            xx,yy = np.meshgrid(np.linspace(-r,r), np.linspace(-r,r))
            ax.plot_surface(xx,xx,yy,alpha=.25)
    return ax

def plot_proj(dataset, kernels,axes=None,cmap='autumn_r',arrow=False): 
    if 'composite' in kernels:
        im = np.mod(dataset[kernels]['time'],0.75) == 0
    else:
        im = np.mod(dataset[kernels]['time'],0.75) < 0.03
    if axes is None:
        fig,axes = plt.subplots(2,3,figsize=(10,5))
    plot_colored_time2(axes[0,0], dataset, kernels, CRE[0], CRE[1], cmap, arrow, im)    
    plot_colored_time2(axes[0,1], dataset, kernels, CRE[0], CRE[2], cmap, arrow, im)         
    plot_colored_time2(axes[0,2], dataset, kernels, CRE[1], CRE[2], cmap, arrow, im)

    #plot_colored_time2(axes[1,0], dataset, kernels, CRE[0], CRE[1], cmap, arrow, im)
    plot_colored_time(axes[1,0],dataset[kernels][CRE[0]]-dataset[kernels][CRE[1]],dataset[kernels][CRE[2]],dataset[kernels]['time'],cmap=cmap,arrow=arrow)
    axes[1,0].scatter(dataset[kernels][CRE[0]][im]-dataset[kernels][CRE[1]][im],dataset[kernels][CRE[2]][im],c=dataset[kernels]['time'][im],cmap=cmap,edgecolor='k')
    axes[1,0].set_xlabel(CRE[0] + '-'+CRE[1])
    axes[1,0].set_ylabel(CRE[2])
          
    plot_colored_time(axes[1,1],dataset[kernels][CRE[0]],dataset[kernels][CRE[1]]-dataset[kernels][CRE[2]],dataset[kernels]['time'],cmap=cmap,arrow=arrow)
    axes[1,1].scatter(dataset[kernels][CRE[0]][im],dataset[kernels][CRE[1]][im]-dataset[kernels][CRE[2]][im],c=dataset[kernels]['time'][im],cmap=cmap,edgecolor='k')
    axes[1,1].set_xlabel(CRE[0])
    axes[1,1].set_ylabel(CRE[1]+ '-'+CRE[2])

    plot_colored_time(axes[1,2],dataset[kernels]['time'],dataset[kernels][CRE[0]],dataset[kernels]['time'],cmap=cmap,arrow=False)
    axes[1,2].scatter(dataset[kernels]['time'][im],dataset[kernels][CRE[0]][im],c=dataset[kernels]['time'][im],cmap=cmap,edgecolor='k')
    axes[1,2].set_xlabel('Time')
    axes[1,2].set_ylabel(CRE[0])
    add_stimulus_bars(axes[1,2], dataset[kernels]['time'],kernels)
    axes[1,2].set_xlim(0, np.max(dataset[kernels]['time']))
    plt.tight_layout()
    return axes

def plot_colored_time2(axes, dataset, kernels, c1, c2, cmap, arrow, im):
    plot_colored_time(axes, dataset[kernels][c1], dataset[kernels][c2],
        dataset[kernels]['time'],cmap=cmap,arrow=arrow)
    #axes.fill_between(dataset[kernels][c1], 
    #    dataset[kernels][c2]-dataset[kernels][c2+'_sem'],  
    #    dataset[kernels][c2]+dataset[kernels][c2+'_sem'],
    #    color='k',alpha=.15,ec=None,interpolate=True)
    #axes.fill_betweenx(dataset[kernels][c2], 
    #    dataset[kernels][c1]-dataset[kernels][c1+'_sem'],  
    #    dataset[kernels][c1]+dataset[kernels][c1+'_sem'],
    #    color='k',alpha=.15,ec=None,interpolate=True)
    axes.scatter(dataset[kernels][c1][im],dataset[kernels][c2][im],
        c=dataset[kernels]['time'][im],cmap=cmap,edgecolor='k')
    axes.set_xlabel(c1)
    axes.set_ylabel(c2)

def plot_colored_time(ax, x,y,time,cmap,arrow):
    points = np.array([x,y]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-1], points[1:]],axis=1)
    norm = plt.Normalize(time[0], time[-1])
    lc = LineCollection(segments, cmap=cmap,norm=norm,linewidths=2,zorder=1)
    lc.set_array(time)
    line = ax.add_collection(lc)
    if arrow:
        add_arrow(lc,len(time)-20,time)
    ax.axvline(0, linestyle='--', color='k',alpha=.25)
    ax.axhline(0, linestyle='--', color='k',alpha=.25)

def plot_colored_time_3D(ax, x,y,z,time):
    points = np.array([x,y,z]).T.reshape(-1,1,3)
    segments = np.concatenate([points[:-1], points[1:]],axis=1)
    norm = plt.Normalize(time[0], time[-1]*1.5)
    lc = LineCollection(segments, cmap='Blues_r',norm=norm)
    lc.set_array(time)
    line = ax.add_collection(lc)

def add_stimulus_bars(ax,time, kernel, alpha=.15):
    times = set(np.arange(0,time[-1],0.75))
    if 'omission' in kernel:
        times.remove(0)
    for t in times:
        ax.axvspan(t,t+.25, color='blue',alpha=alpha, zorder=-np.inf)

def add_arrow(line, index,time, size=30):
    c = time[index]/time[-1]
    cmap = line.get_cmap()
    data = line.get_segments()
    line.axes.annotate('',
        xytext=(data[index][0,0],data[index][0,1]),
        xy=(data[index][1,0],data[index][1,1]),
        arrowprops=dict(arrowstyle='->',color=cmap(c),linewidth=2,alpha=0.8),
        size=size)

def save_figure(fig,model_version,filename):
    glm_dir = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/'
    if not os.path.isdir(glm_dir + 'v_'+model_version +'/figures/population'):
        os.mkdir(glm_dir + 'v_'+model_version +'/figures/population')
    plt.savefig(glm_dir + 'v_'+ model_version +'/figures/population/'+filename+".svg")
    plt.savefig(glm_dir + 'v_'+ model_version +'/figures/population/'+filename+".png")
    print(filename)
