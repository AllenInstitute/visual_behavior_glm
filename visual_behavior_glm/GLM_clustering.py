import numpy as np
import pandas as pd
from scipy.stats import chisquare
import matplotlib.pyplot as plt
import visual_behavior.data_access.loading as loading
from mpl_toolkits.axes_grid1 import make_axes_locatable

filedir = '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/v_24_events_all_L2_optimize_by_session/figures/clustering/'

def final(df, cre):
    '''
        Returns two tables
        proportion_table contains the proportion of cells in each depth/area found in each cluster, relative to the average proportion across depth/areas for that cluster

        stats_table returns statistical tests on the proportion of cells in each depth/area
        Use 'bh_significant' unless you have a good reason to use the uncorrected tests
    '''
    proportion_table = compute_cluster_proportion_cre(df, cre)
    stats_table = stats(df,cre)
    return proportion_table, stats_table

def cluster_frequencies():
    '''
        Generates 4 differn plots of clustering frequency/proportion analysis
        1. The proportions of each location (depth/area) in each cluster
        2. The proportions of each location (depth/area) in each cluster 
           relative to "chance" of 1/n-clusters (evenly distributed cells across clusters)
        3. The proportions of each location (depth/area) in each cluster
           relative to the average proportion across locations in that cluster 
           (clusters have the same proportion across locations)
        4. The proportion of each location in each cluster
           relative to the average proportion across locations in that cluster
           but using a multiplicative perspective instead of a linear perspective. 
    '''
    df = load_cluster_labels()
    plot_proportions(df)
    plot_proportion_differences(df)
    plot_cluster_proportions(df)
    plot_cluster_percentages(df)   
 
def load_cluster_labels():

    # Load cluster labels
    filepath = '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/platform_paper_plots/figure_4/24_events_all_L2_optimize_by_session/220223/cluster_ids_Slc17a7_10_Sst_6_Vip_12.hdf'
    df = pd.read_hdf(filepath, key='df')

    # load cell data
    cells_table = loading.get_cell_table(platform_paper_only=True).reset_index().drop_duplicates(subset='cell_specimen_id')
    df = df.drop(columns=['labels','cre_line'])
    df = pd.merge(df, cells_table, on='cell_specimen_id',suffixes=('','_y'))  

    # Bin depths and annotate
    df['coarse_binned_depth'] = ['upper' if x < 250 else 'lower' for x in df['imaging_depth']]
    df['location'] = df['targeted_structure']+'_'+df['coarse_binned_depth']

    # Remove clusters with less than 5 cells
    df = df.drop(df.index[(df['cre_line']=="Sst-IRES-Cre")&(df['cluster_id']==6)])
    df = df.drop(df.index[(df['cre_line']=="Slc17a7-IRES2-Cre")&(df['cluster_id']==10)])

    return df

def plot_proportions(df):
    fig, ax = plt.subplots(1,3,figsize=(8,4))
    fig.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.9, wspace=1)
    plot_proportion_cre(df, fig, ax[2], 'Slc17a7-IRES2-Cre')
    plot_proportion_cre(df, fig, ax[1], 'Sst-IRES-Cre')
    plot_proportion_cre(df, fig, ax[0], 'Vip-IRES-Cre')
    plt.savefig(filedir+'cluster_proportions.svg')
    plt.savefig(filedir+'cluster_proportions.png')

def compute_proportion_cre(df, cre):
    '''
        Computes the proportion of cells in each cluster within each location
    '''
    # Count cells in each area/cluster
    table = df.query('cre_line == @cre').groupby(['cluster_id','location'])['cell_specimen_id'].count().unstack()
    table = table[['VISp_upper','VISp_lower','VISl_upper','VISl_lower']]
    table = table.fillna(value=0)

    # compute fraction in each area/cluster
    depth_areas = table.columns.values
    for da in depth_areas:
        table[da] = table[da]/table[da].sum()
    return table

def plot_proportion_cre(df,fig,ax, cre):
    '''
        Fraction of cells per area&depth 
    '''
    table = compute_proportion_cre(df, cre)

    # plot proportions
    cbar = ax.imshow(table,cmap='Purples')
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(table.columns.values,rotation=90)
    ax.set_ylabel('Cluster #',fontsize=16)  
    ax.set_title(mapper(cre),fontsize=16) 
    fig.colorbar(cbar, ax=ax,label='fraction of cells per area & depth')
   
    # Add statistics 
    table2 = stats(df, cre)
    for index in table2.index.values:
        if table2.loc[index]['bh_significant']:
            ax.plot(-1,index-1,'r*')
    ax.set_xlim(-1.5,3.5)
    ax.set_xticks([-1,0,1,2,3])
    ax.set_xticklabels(np.concatenate([['p<0.05'],table.columns.values]),rotation=90)
    ax.axvline(-0.5,color='k',linewidth=.5)

def plot_proportion_differences(df):
    fig, ax = plt.subplots(1,3,figsize=(8,4))
    fig.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.9, wspace=1)
    plot_proportion_differences_cre(df, fig, ax[2], 'Slc17a7-IRES2-Cre')
    plot_proportion_differences_cre(df, fig, ax[1], 'Sst-IRES-Cre')
    plot_proportion_differences_cre(df, fig, ax[0], 'Vip-IRES-Cre')
    plt.savefig(filedir+'cluster_proportion_differences.svg')
    plt.savefig(filedir+'cluster_proportion_differences.png')

def compute_proportion_differences_cre(df, cre):
    # count cells in each area/cluster
    table = df.query('cre_line == @cre').groupby(['cluster_id','location'])['cell_specimen_id'].count().unstack()
    table = table[['VISp_upper','VISp_lower','VISl_upper','VISl_lower']]
    table = table.fillna(value=0)
    nclusters = len(table.index.values)

    # compute fraction in each area relative to expected fraction
    depth_areas = table.columns.values
    for da in depth_areas:
        table[da] = table[da]/table[da].sum() - 1/nclusters
    return table

def plot_proportion_differences_cre(df,fig,ax, cre):
    '''
        Fraction of cells per area & depth, then
        subtract expected fraction (1/n)
    '''

    table = compute_proportion_differences_cre(df,cre)

    # plot fractions
    vmax = table.abs().max().max()
    cbar = ax.imshow(table,cmap='PRGn',vmin=-vmax,vmax=vmax)
    fig.colorbar(cbar, ax=ax,label='fraction of cells per area & depth \nrelative to evenly distributed across clusters')
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(table.columns.values,rotation=90)
    ax.set_ylabel('Cluster #',fontsize=16)  
    ax.set_title(mapper(cre),fontsize=16) 

def plot_cluster_proportions(df):
    fig, ax = plt.subplots(1,3,figsize=(8,4))
    fig.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.9, wspace=1)
    plot_cluster_proportion_cre(df, fig, ax[2], 'Slc17a7-IRES2-Cre')
    plot_cluster_proportion_cre(df, fig, ax[1], 'Sst-IRES-Cre')
    plot_cluster_proportion_cre(df, fig, ax[0], 'Vip-IRES-Cre')
    plt.savefig(filedir+'within_cluster_proportions.svg')
    plt.savefig(filedir+'within_cluster_proportions.png')

def compute_cluster_proportion_cre(df, cre):
    table = compute_proportion_cre(df, cre)

    # get average proportion in each cluster
    table['mean'] = table.mean(axis=1)

    # compute proportion in each area relative to cluster average
    depth_areas = table.columns.values
    for da in depth_areas:
        table[da] = table[da] - table['mean'] 

    # plot proportions
    table = table[['VISp_upper','VISp_lower','VISl_upper','VISl_lower']]
    return table

def plot_cluster_proportion_cre(df,fig,ax, cre):
    '''
        Fraction of cells per area&depth 
    '''
    table = compute_cluster_proportion_cre(df,cre)

    vmax = table.abs().max().max()
    vmax = .15
    cbar = ax.imshow(table,cmap='PRGn',vmin=-vmax, vmax=vmax)
    fig.colorbar(cbar, ax=ax,label='proportion of cells per area & depth \n relative to cluster average')
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(table.columns.values,rotation=90)
    ax.set_ylabel('Cluster #',fontsize=16)  
    ax.set_title(mapper(cre),fontsize=16) 
    
    # add statistics
    table2 = stats(df, cre)
    for index in table2.index.values:
        if table2.loc[index]['bh_significant']:
            ax.plot(-1,index-1,'r*')
    ax.set_xlim(-1.5,3.5)
    ax.set_xticks([-1,0,1,2,3])
    ax.set_xticklabels(np.concatenate([['p<0.05'],table.columns.values]),rotation=90)
    ax.axvline(-0.5,color='k',linewidth=.5)

def plot_cluster_percentages(df):
    fig, ax = plt.subplots(1,3,figsize=(8,4))
    fig.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.9, wspace=1)
    plot_cluster_percentage_cre(df, fig, ax[2], 'Slc17a7-IRES2-Cre')
    plot_cluster_percentage_cre(df, fig, ax[1], 'Sst-IRES-Cre')
    plot_cluster_percentage_cre(df, fig, ax[0], 'Vip-IRES-Cre')
    plt.savefig(filedir+'within_cluster_percentages.svg')
    plt.savefig(filedir+'within_cluster_percentages.png')

def plot_cluster_percentage_cre(df,fig,ax, cre):
    '''
        Fraction of cells per area&depth 
    '''
    # count cells in each area/cluster
    table = df.query('cre_line == @cre').groupby(['cluster_id','location'])['cell_specimen_id'].count().unstack()
    table = table[['VISp_upper','VISp_lower','VISl_upper','VISl_lower']]
    table = table.fillna(value=0)

    # compute proportion in each area/cluster
    depth_areas = table.columns.values
    for da in depth_areas:
        table[da] = table[da]/table[da].sum()

    # get average proportion in each cluster
    table['mean'] = table.mean(axis=1)

    # build second table with cells in each area/cluster
    table2 = df.query('cre_line == @cre').groupby(['cluster_id','location'])['cell_specimen_id'].count().unstack()
    table2 = table2[['VISp_upper','VISp_lower','VISl_upper','VISl_lower']]
    table2 = table2.fillna(value=0)

    # estimate chance cell counts based on average proportion in each cluster
    # then add relative fraction of actual counts compared to chance counts
    # subtract 1 so 0=chance
    for da in depth_areas:
        table2[da+'_chance_count'] = table2[da].sum()*table['mean']
        table2[da+'_rel_fraction'] = table2[da]/table2[da+'_chance_count']-1
    
    # plot proportions 
    table2 = table2[['VISp_upper_rel_fraction','VISp_lower_rel_fraction','VISl_upper_rel_fraction','VISl_lower_rel_fraction']]
    cbar = ax.imshow(table2,cmap='PRGn',vmin=-1,vmax=1)
    fig.colorbar(cbar, ax=ax,label='fraction of cells per area & depth')
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(table.columns.values[0:4],rotation=90)
    ax.set_ylabel('Cluster #',fontsize=16)  
    ax.set_title(mapper(cre),fontsize=16)
    
    # add statistics
    table2 = stats(df, cre)
    for index in table2.index.values:
        if table2.loc[index]['bh_significant']:
            ax.plot(-1,index-1,'r*')
    ax.set_xlim(-1.5,3.5)
    ax.set_xticks([-1,0,1,2,3])
    ax.set_xticklabels(np.concatenate([['p<0.05'],table.columns.values[0:4]]),rotation=90)
    ax.axvline(-0.5,color='k',linewidth=.5)

def stats(df,cre):
    '''
        Performs chi-squared tests to asses whether the observed cell counts in each area/depth differ
        significantly from the average for that cluster. 
    '''    

    # compute cell counts in each area/cluster
    table = df.query('cre_line == @cre').groupby(['cluster_id','location'])['cell_specimen_id'].count().unstack()
    table = table[['VISp_upper','VISp_lower','VISl_upper','VISl_lower']]
    table = table.fillna(value=0)
    depth_areas = table.columns.values

    # compute proportion for null hypothesis that areas have the same proportions
    table['total_cells'] = table.sum(axis=1)
    table['null_mean_proportion'] = table['total_cells']/np.sum(table['total_cells'])

    # second table of cell counts in each area/cluster
    table2 = df.query('cre_line == @cre').groupby(['cluster_id','location'])['cell_specimen_id'].count().unstack()
    table2 = table2[['VISp_upper','VISp_lower','VISl_upper','VISl_lower']]
    table2 = table2.fillna(value=0)

    # compute estimated frequency of cells based on average fraction for each cluster
    for da in depth_areas:
        table2[da+'_chance_count'] = table2[da].sum()*table['null_mean_proportion']

    # perform chi-squared test
    for index in table2.index.values:
        f = table2.loc[index][['VISp_upper','VISp_lower','VISl_upper','VISl_lower']].values
        f_expected = table2.loc[index][['VISp_upper_chance_count','VISp_lower_chance_count','VISl_upper_chance_count','VISl_lower_chance_count']].values
        
        # Manually doing check here bc Im on old version of scipy
        assert np.abs(np.sum(f) - np.sum(f_expected))<1, 'f and f_expected must be the same'
        out = chisquare(f,f_expected)
        table2.at[index, 'pvalue'] = out.pvalue
        table2.at[index, 'significant'] = out.pvalue < 0.05

    # Use Benjamini Hochberg Correction for multiple comparisons
    table2 = add_hochberg_correction(table2)
    return table2

def mapper(cre):
    mapper = {
        'Slc17a7-IRES2-Cre':'Excitatory',
        'Sst-IRES-Cre':'Sst',
        'Vip-IRES-Cre':'Vip',
        }
    return mapper[cre]

def add_hochberg_correction(table):
    '''
        Performs the Benjamini Hochberg correction
    '''    
    # Sort table by pvalues
    table = table.sort_values(by='pvalue').reset_index()
    
    # compute the corrected pvalue based on the rank of each test
    # Need to use rank starting at 1
    table['imq'] = (1+table.index.values)/len(table)*0.05

    # Find the largest pvalue less than its corrected pvalue
    # all tests above that are significant
    table['bh_significant'] = False
    passing_tests = table[table['pvalue'] < table['imq']]
    if len(passing_tests) >0:
        last_index = table[table['pvalue'] < table['imq']].tail(1).index.values[0]
        table.at[last_index,'bh_significant'] = True
        table.at[0:last_index,'bh_significant'] = True
    
    # reset order of table and return
    return table.sort_values(by='cluster_id').set_index('cluster_id') 
 

