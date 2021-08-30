import os
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib
from matplotlib import cm
from matplotlib.collections import LineCollection

CRE = ['Slc17a7-IRES2-Cre','Sst-IRES-Cre','Vip-IRES-Cre']
GLM_VERSION = '12_dff_L2_optimize_by_session'
# TODO
# direction arrows on traces


def get_dataset(image_set='familiar'):
    import visual_behavior_modeling.utils as utils
    dataset = utils.make_dataset(image_set=image_set)
    
    trim = ['composite_images','composite_omission','composite_change']
    for t in trim:
        for k in dataset[t]:
            if k != 'meta':
                dataset[t][k] = dataset[t][k][96:]
            if k == 'time':
                dataset[t][k] = dataset[t][k]-3
    return dataset

def compare_against(dataset_f,dataset_n):
    ax = plot_3D(dataset_n, 'composite_omission', color='b')
    ax = plot_3D(dataset_n, 'composite_images',ax=ax,color='m')

    ax = plot_proj(dataset_n, 'composite_omission', cmap='winter_r',arrow=False)
    ax = plot_proj(dataset_n, 'composite_images',axes=ax,arrow=True,cmap='cool')

    ax = plot_3D(dataset_f, 'composite_omission', color='r')
    ax = plot_3D(dataset_f, 'composite_images',ax=ax,color='m')

    ax = plot_proj(dataset_f, 'composite_omission', arrow=False)
    ax = plot_proj(dataset_f, 'composite_images',axes=ax,arrow=True,cmap='cool')


def compare_images(dataset_f, dataset_n):
    # Images
    ax = plot_3D(dataset_n, 'composite_images',color='b')
    save_figure(plt.gcf(),GLM_VERSION, 'novel_image_3d') 
    ax = plot_3D(dataset_f, 'composite_images')
    save_figure(plt.gcf(),GLM_VERSION, 'familiar_image_3d') 
    ax = plot_3D(dataset_n, 'composite_images',ax=ax, color='b')
    save_figure(plt.gcf(),GLM_VERSION, 'compare_image_3d') 

    # Comparing familiar/novel images
    ax = plot_proj(dataset_n, 'composite_images',arrow=True,cmap='winter_r')
    save_figure(plt.gcf(),GLM_VERSION, 'novel_image_proj') 
    ax = plot_proj(dataset_f, 'composite_images',arrow=True)
    save_figure(plt.gcf(),GLM_VERSION, 'familiar_image_proj') 
    ax = plot_proj(dataset_n, 'composite_images',axes=ax, cmap='winter_r',arrow=True)
    save_figure(plt.gcf(),GLM_VERSION, 'compare_image_proj') 

    #omissions
    ax = plot_3D(dataset_n, 'composite_omission', color='b')
    save_figure(plt.gcf(),GLM_VERSION, 'novel_omission_3d') 
    ax = plot_3D(dataset_f, 'composite_omission')
    save_figure(plt.gcf(),GLM_VERSION, 'familiar_omission_3d') 
    ax = plot_3D(dataset_n, 'composite_omission',ax=ax, color='b')
    save_figure(plt.gcf(),GLM_VERSION, 'compare_omission_3d') 

    # Comparing familar/novel omissions
    ax = plot_proj(dataset_n, 'composite_omission', cmap='winter_r',arrow=False)
    save_figure(plt.gcf(),GLM_VERSION, 'novel_omission_proj') 
    ax = plot_proj(dataset_f, 'composite_omission',arrow=False)
    save_figure(plt.gcf(),GLM_VERSION, 'familiar_omission_proj') 
    ax = plot_proj(dataset_n, 'composite_omission',axes=ax, cmap='winter_r',arrow=False)
    save_figure(plt.gcf(),GLM_VERSION, 'compare_omission_proj') 

    # Change
    ax = plot_3D(dataset_n, 'composite_change', color='b')
    save_figure(plt.gcf(),GLM_VERSION, 'novel_change_3d') 
    ax = plot_3D(dataset_f, 'composite_change')
    save_figure(plt.gcf(),GLM_VERSION, 'familiar_change_3d') 
    ax = plot_3D(dataset_n, 'composite_change',ax=ax, color='b')
    save_figure(plt.gcf(),GLM_VERSION, 'compare_change_3d') 

    # Comparing familiar/novel change
    ax = plot_proj(dataset_n, 'composite_change', cmap='winter_r',arrow=False)
    save_figure(plt.gcf(),GLM_VERSION, 'novel_change_proj') 
    ax = plot_proj(dataset_f, 'composite_change',arrow=False)
    save_figure(plt.gcf(),GLM_VERSION, 'familiar_change_proj') 
    ax = plot_proj(dataset_n, 'composite_change',axes=ax, cmap='winter_r',arrow=False)
    save_figure(plt.gcf(),GLM_VERSION, 'compare_change_proj') 

def plot_3D(dataset, kernels,ax=None,color='r'):
    add_box=False
    if ax is None:
        add_box=True
        fig = plt.figure()
        ax = plt.axes(projection='3d')
    line = ax.plot3D(dataset[kernels][CRE[0]],dataset[kernels][CRE[1]],dataset[kernels][CRE[2]],linewidth=2,color=color)
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
            xx,yy = np.meshgrid(np.linspace(-1,1.5), np.linspace(-1,8))
            ax.plot_surface(xx,xx,yy,alpha=.25)
    else:
        ax.set_xlim(-1,1)    
        ax.set_ylim(-1,1)
        ax.set_zlim(-1,1) 
        ax.view_init(elev=35, azim=-45)
        if add_box:
            xx,yy = np.meshgrid(np.linspace(-1,1), np.linspace(-1,1))
            ax.plot_surface(xx,xx,yy,alpha=.25)
    return ax

def plot_proj(dataset, kernels,axes=None,cmap='autumn_r',arrow=False): 
    if 'composite' in kernels:
        im = np.mod(dataset[kernels]['time'],0.75) == 0
    else:
        im = np.mod(dataset[kernels]['time'],0.75) < 0.03
    if axes is None:
        fig,axes = plt.subplots(2,3,figsize=(10,5))
    plot_colored_time(axes[0,0], dataset[kernels][CRE[0]], dataset[kernels][CRE[1]],dataset[kernels]['time'],cmap=cmap,arrow=arrow)
    axes[0,0].scatter(dataset[kernels][CRE[0]][im],dataset[kernels][CRE[1]][im],c=dataset[kernels]['time'][im],cmap=cmap,edgecolor='k')
    axes[0,0].set_xlabel(CRE[0])
    axes[0,0].set_ylabel(CRE[1])
    axes[0,0].axline((1,1), slope=1,ls='--',alpha=.25,color='k')
    plt.tight_layout()  
      
    plot_colored_time(axes[0,1],dataset[kernels][CRE[0]],dataset[kernels][CRE[2]],dataset[kernels]['time'],cmap=cmap,arrow=arrow)
    axes[0,1].scatter(dataset[kernels][CRE[0]][im],dataset[kernels][CRE[2]][im],c=dataset[kernels]['time'][im],cmap=cmap,edgecolor='k')
    axes[0,1].set_xlabel(CRE[0])
    axes[0,1].set_ylabel(CRE[2])
          
    plot_colored_time(axes[0,2],dataset[kernels][CRE[1]],dataset[kernels][CRE[2]],dataset[kernels]['time'],cmap=cmap,arrow=arrow)
    axes[0,2].scatter(dataset[kernels][CRE[1]][im],dataset[kernels][CRE[2]][im],c=dataset[kernels]['time'][im],cmap=cmap,edgecolor='k')
    axes[0,2].set_xlabel(CRE[1])
    axes[0,2].set_ylabel(CRE[2])

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
