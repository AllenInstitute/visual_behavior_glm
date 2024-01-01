import numpy as np
import pandas as pd
from scipy.stats import chisquare
from scipy.stats import chi2_contingency
from scipy.stats import power_divergence
from scipy.stats import fisher_exact
# import FisherExact (Used for non2x2 tables of Fisher Exact test, not used but leaving a note)
import matplotlib.pyplot as plt
import visual_behavior.data_access.loading as loading
from mpl_toolkits.axes_grid1 import make_axes_locatable

filedir = '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/v_24_events_all_L2_optimize_by_session/figures/clustering/'


def compare_stats(num_shuffles=1000):
    plt.figure()
    for i in np.arange(100, 1000, 10):
        stats = compare_shuffle(n=i, num_shuffles=num_shuffles)
        pval = stats[0]
        chi = stats[1]
        if i == 100:
            plt.plot(i, chi, 'bo', label='chi-2')
            plt.plot(i, pval, 'ro', label='distribution')
        else:
            plt.plot(i, chi, 'bo')
            plt.plot(i, pval, 'ro')

    plt.axhline(.05, linestyle='--', color='k')
    plt.legend()
    plt.ylabel('p value')
    plt.xlabel('num cells')
    plt.ylim(0, .5)


<<<<<<< Updated upstream
def compare_shuffle(n=100, p=.15, pn=.1, num_shuffles=1000):
=======

def compare_shuffle(n=100,p=.15,pn=.1,num_shuffles=1000):  
>>>>>>> Stashed changes
    # worried about independence
    num_h = int(np.floor(pn*n))
    num_m = n-num_h
    raw = [1]*(num_h)+[0]*(num_m)

    # Generate shuffle
    num_hits = []
    for i in np.arange(0, num_shuffles):
        #shuffle = np.random.rand(n) < pn
        shuffle = np.random.choice(raw, n)
        num_hits.append(np.sum(shuffle))

    # Compute chi-square where the data is 15%/85% of cells
    # and null is the mean % across shuffles
    data = [p*n*1000000000, (1-p)*n*1000000000]
    null = [np.mean(num_hits), n-np.mean(num_hits)]
    x = np.floor(np.array([data, null])).T
    out = chi2_contingency(x, correction=True)

    # Compare that with a p-value where we ask what percentage of the shuffles
    # had more than 15%
    pval = np.sum(np.array(num_hits) >= p*n)/num_shuffles*2

    return pval, out[1]


<<<<<<< Updated upstream
def final(df, cre='none', areas=None, test='chi_squared_'):
=======
def final(df, cre, locations=None, test='chi_squared_'):
>>>>>>> Stashed changes
    '''
        Returns two tables
        proportion_table contains the proportion of cells in each location found in each cluster, relative to the average proportion across location for that cluster
        stats_table returns statistical tests on the proportion of cells in each location 

        Assumes that df has a column called 'location' that contains categorical variables to compute proportions over

        Use 'bh_significant' unless you have a good reason to use the uncorrected tests
    '''
<<<<<<< Updated upstream
    if areas is None:
        # Get areas
        areas = np.sort(df['location'].unique())
    else:
        assert set(areas) == set(df['location'].unique(
        )), "areas passed in don't match location column"

    proportion_table = compute_cluster_proportion_cre(df, cre=cre, areas=areas)
    stats_table = stats(df, cre=cre, areas=areas, test=test)
=======
    if locations is None:
        # Get locations
        locations = np.sort(df['location'].unique())   
    else:
        assert set(locations) == set(df['location'].unique()), "locations passed in don't match location column" 

    proportion_table = compute_cluster_proportion_cre(df, cre, locations)
    stats_table = stats(df, cre, locations, test=test)

>>>>>>> Stashed changes
    return proportion_table, stats_table


def cluster_frequencies():
    '''
        Generates 4 different plots of clustering frequency/proportion analysis
        1. The proportions of each location in each cluster
        2. The proportions of each location in each cluster 
           relative to "chance" of 1/n-clusters (evenly distributed cells across clusters)
        3. The proportions of each location in each cluster
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
<<<<<<< Updated upstream
    plot_cluster_percentages(df)


=======
    plot_cluster_percentages(df)   

 
>>>>>>> Stashed changes
def load_cluster_labels():
    '''
        - Loads a dataframe of cluster labels
        - merges in cell table data 
        - defines a `location` column with depth/location combinations
        - drops clusters with less than 5 cells
    '''

    # Load cluster labels
    filepath = '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/platform_paper_plots/figure_4/24_events_all_L2_optimize_by_session/220622_across_session_norm_10_5_10/cluster_labels_Vip_10_Sst_5_Slc17a7_10.h5'
    df = pd.read_hdf(filepath, key='df')

    # load cell data
    cells_table = loading.get_cell_table(platform_paper_only=True, include_4x2_data=False).reset_index(
    ).drop_duplicates(subset='cell_specimen_id')
    df = df.drop(columns=['labels', 'cre_line'])
    df = pd.merge(df, cells_table, on='cell_specimen_id', suffixes=('', '_y'))

    # Bin depths and annotate
    df['coarse_binned_depth'] = ['upper' if x <
                                 250 else 'lower' for x in df['imaging_depth']]
    df['location'] = df['targeted_structure']+'_'+df['coarse_binned_depth']

    # Remove clusters with less than 5 cells
    #df = df.drop(df.index[(df['cre_line']=="Sst-IRES-Cre")&(df['cluster_id']==6)])
    #df = df.drop(df.index[(df['cre_line']=="Slc17a7-IRES2-Cre")&(df['cluster_id']==10)])

    return df


<<<<<<< Updated upstream
def plot_proportions(df, areas=None, savefig=False, extra='', test='chi_squared_'):
=======
def plot_proportions(df, locations=None, savefig=False, extra='', test='chi_squared_'):
>>>>>>> Stashed changes
    '''
        Compute, then plot, the proportion of cells in each location within each cluster
        Assumes df has 'location' column
    '''
<<<<<<< Updated upstream

    if areas is None:
        # Get areas
        areas = np.sort(df['location'].unique())
    else:
        assert set(areas) == set(df['location'].unique(
        )), "areas passed in don't match location column"
=======
    
    if locations is None:
        # Get locations from locations column
        locations = np.sort(df['location'].unique())   
    else:
        assert set(locations) == set(df['location'].unique()), "locations passed in don't match location column" 
>>>>>>> Stashed changes

    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    fig.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.9, wspace=1)
<<<<<<< Updated upstream
    plot_proportion_cre(df, areas, fig, ax[2], 'Slc17a7-IRES2-Cre', test=test)
    plot_proportion_cre(df, areas,  fig, ax[1], 'Sst-IRES-Cre', test=test)
    plot_proportion_cre(df, areas,  fig, ax[0], 'Vip-IRES-Cre', test=test)
=======
    plot_proportion_cre(df, locations, fig, ax[2], 'Slc17a7-IRES2-Cre',test=test)
    plot_proportion_cre(df, locations,  fig, ax[1], 'Sst-IRES-Cre',test=test)
    plot_proportion_cre(df, locations,  fig, ax[0], 'Vip-IRES-Cre',test=test)
>>>>>>> Stashed changes
    if savefig:
        extra = extra+'_'+test
        plt.savefig(filedir+'cluster_proportions'+extra+'.svg')
        plt.savefig(filedir+'cluster_proportions'+extra+'.png')


<<<<<<< Updated upstream
def compute_proportion_cre(df, cre='none', areas=[]):
=======
def compute_proportion_cre(df, cre, locations=None):
>>>>>>> Stashed changes
    '''
        Computes the proportion of cells in each cluster within each location

        location must be a column of string names. The statistics are computed relative to whatever the location column contains
    '''
    # Get locations from locations column if not provided
    if locations is None:
        # Get locations from locations column
        locations = np.sort(df['location'].unique())   
    else:
        assert set(locations) == set(df['location'].unique()), "locations passed in don't match location column" 

<<<<<<< Updated upstream
    # Count cells in each area/cluster
    if cre is 'none':
        table = df.groupby(['cluster_id', 'location'])[
            'cell_specimen_id'].count().unstack()
    else:
        table = df.query('cre_line == @cre').groupby(['cluster_id', 'location'])[
            'cell_specimen_id'].count().unstack()
    table = table[areas]
=======
    # Count cells in each location/cluster
    table = df.query('cre_line == @cre').groupby(['cluster_id', 'location'])['cell_specimen_id'].count().unstack()
    table = table[locations]
>>>>>>> Stashed changes
    table = table.fillna(value=0)

    # compute fraction in each location/cluster
    for a in locations:
        table[a] = table[a]/table[a].sum() # number of cells per location/cluster divided by total number of cells per location
    return table


<<<<<<< Updated upstream
def plot_proportion_cre(df, areas, fig, ax, cre='none', test='chi_squared_'):
=======
def plot_proportion_cre(df, locations,  fig, ax, cre, test='chi_squared_'):
>>>>>>> Stashed changes
    '''
        Fraction of cells per location&depth 
    '''

    # Get proportions
<<<<<<< Updated upstream
    table = compute_proportion_cre(df, cre=cre, areas=areas)

    # plot proportions
    cbar = ax.imshow(table, cmap='Purples', vmax=.4)
    ax.set_ylabel('Cluster #', fontsize=16)
    ax.set_yticks(range(0, len(table)))
    ax.set_yticklabels(table.index.values+1, size=16)
    ax.set_title(mapper(cre), fontsize=16)
    fig.colorbar(cbar, ax=ax, label='fraction of cells per location')

    # Add statistics
    table2 = stats(df, cre=cre, areas=areas, test=test)
=======
    table = compute_proportion_cre(df, cre, locations)

    # plot proportions
    cbar = ax.imshow(table,cmap='Purples',vmax=.4)
    ax.set_ylabel('Cluster #',fontsize=16)  
    ax.set_yticks(range(0,len(table)))
    ax.set_yticklabels(range(1,len(table)+1))
    ax.set_title(mapper(cre),fontsize=16) 
    fig.colorbar(cbar, ax=ax,label='fraction of cells per location')
   
    # Add statistics 
    table2 = stats(df, cre,locations,test=test)
>>>>>>> Stashed changes
    for index in table2.index.values:
        if table2.loc[index]['bh_significant']:
            ax.plot(-1, index, 'r*')

<<<<<<< Updated upstream
    num_areas = len(areas)
    ax.set_xlim(-1.5, num_areas-.5)
    ax.set_xticks(range(-1, num_areas))
    ax.set_xticklabels(np.concatenate([[test[:-1]], areas]), rotation=90)
    ax.axvline(-0.5, color='k', linewidth=.5)


def plot_proportion_differences(df, areas=None):
=======
    num_locations = len(locations)
    ax.set_xlim(-1.5,num_locations-.5)
    ax.set_xticks(range(-1,num_locations))
    ax.set_xticklabels(np.concatenate([[test[:-1]],locations]),rotation=90)
    ax.axvline(-0.5,color='k',linewidth=.5)


def plot_proportion_differences(df, locations=None):
>>>>>>> Stashed changes
    '''
        Computes, then plots, the proportion of cells in each location within each cluster
        relative to a 1/n average distribution across n clusters. 
    '''

<<<<<<< Updated upstream
    if areas is None:
        # Get areas
        areas = np.sort(df['location'].unique())
        areas = areas[::-1]
    else:
        assert set(areas) == set(df['location'].unique(
        )), "areas passed in don't match location column"
=======
    if locations is None:
        # Get locations
        locations = np.sort(df['location'].unique())    
    else:
        assert set(locations) == set(df['location'].unique()), "locations passed in don't match location column" 
>>>>>>> Stashed changes

    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    fig.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.9, wspace=1)
    plot_proportion_differences_cre(df, locations, fig, ax[2], 'Slc17a7-IRES2-Cre')
    plot_proportion_differences_cre(df, locations, fig, ax[1], 'Sst-IRES-Cre')
    plot_proportion_differences_cre(df, locations, fig, ax[0], 'Vip-IRES-Cre')
    plt.savefig(filedir+'cluster_proportion_differences.svg')
    plt.savefig(filedir+'cluster_proportion_differences.png')


<<<<<<< Updated upstream
def compute_proportion_differences_cre(df, cre='none', areas=[]):
    '''
        compute proportion differences relative to 1/n average
    '''
    # count cells in each area/cluster
    if cre is 'none':
        table = df.groupby(['cluster_id', 'location'])[
            'cell_specimen_id'].count().unstack()
    else:
        table = df.query('cre_line == @cre').groupby(['cluster_id', 'location'])[
            'cell_specimen_id'].count().unstack()
    table = table[areas]
    table = table.fillna(value=0)
    nclusters = len(table.index.values)

    # compute fraction in each area relative to expected fraction
    for a in areas:
=======
def compute_proportion_differences_cre(df, cre, locations=None):
    '''
        compute proportion differences relative to 1/n average
    '''

    if locations is None:
        # Get locations
        locations = np.sort(df['location'].unique())    
    else:
        assert set(locations) == set(df['location'].unique()), "locations passed in don't match location column" 

    # count cells in each location/cluster
    table = df.query('cre_line == @cre').groupby(['cluster_id','location'])['cell_specimen_id'].count().unstack()
    table = table[locations]
    table = table.fillna(value=0)
    nclusters = len(table.index.values)

    # compute fraction in each location relative to expected fraction 
    for a in locations:
>>>>>>> Stashed changes
        table[a] = table[a]/table[a].sum() - 1/nclusters
    return table


<<<<<<< Updated upstream
def plot_proportion_differences_cre(df, areas, fig, ax, cre='none'):
=======
def plot_proportion_differences_cre(df, locations, fig, ax, cre):
>>>>>>> Stashed changes
    '''
        Fraction of cells per location, then
        subtract expected fraction (1/n)
    '''
<<<<<<< Updated upstream
    table = compute_proportion_differences_cre(df, cre=cre, areas=areas)

    # plot fractions
    vmax = table.abs().max().max()
    cbar = ax.imshow(table[areas], cmap='PRGn', vmin=-vmax, vmax=vmax)
    fig.colorbar(
        cbar, ax=ax, label='fraction of cells per location \nrelative to evenly distributed across clusters')
    ax.set_xticks(range(0, len(areas)))
    ax.set_xticklabels(areas, rotation=90)
    ax.set_ylabel('Cluster #', fontsize=16)
    ax.set_title(mapper(cre), fontsize=16)


def plot_cluster_proportions(df, areas=None):
    if areas is None:
        # Get areas
        areas = np.sort(df['location'].unique())
    else:
        assert set(areas) == set(df['location'].unique(
        )), "areas passed in don't match location column"
=======
    table = compute_proportion_differences_cre(df, cre, locations)

    # plot fractions
    vmax = table.abs().max().max()
    cbar = ax.imshow(table, cmap='PRGn', vmin=-vmax, vmax=vmax)
    fig.colorbar(cbar, ax=ax, label='fraction of cells per location \nrelative to evenly distributed across clusters')
    ax.set_xticks(range(0, len(locations)))
    ax.set_xticklabels(locations, rotation=90)
    ax.set_ylabel('Cluster #', fontsize=16)  
    ax.set_title(mapper(cre), fontsize=16) 


def plot_cluster_proportions(df, locations=None):
    if locations is None:
        # Get locations
        locations = np.sort(df['location'].unique())    
    else:
        assert set(locations) == set(df['location'].unique()), "locations passed in don't match location column" 
>>>>>>> Stashed changes

    fig, ax = plt.subplots(1, 3, figsize=(10, 8))
    fig.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.9, wspace=1)
<<<<<<< Updated upstream
    plot_cluster_proportion_cre(df, areas, fig, ax[2], 'Slc17a7-IRES2-Cre')
    plot_cluster_proportion_cre(df, areas, fig, ax[1], 'Sst-IRES-Cre')
    plot_cluster_proportion_cre(df, areas, fig, ax[0], 'Vip-IRES-Cre')
    plt.savefig(filedir+'within_cluster_proportions.svg')
    plt.savefig(filedir+'within_cluster_proportions.png')
    plt.savefig(filedir + 'within_cluster_proportions.pdf')


def compute_cluster_proportion_cre(df, cre='none', areas=[]):
    table = compute_proportion_cre(df, cre=cre, areas=areas)
=======
    plot_cluster_proportion_cre(df,locations, fig, ax[2], 'Slc17a7-IRES2-Cre')
    plot_cluster_proportion_cre(df,locations, fig, ax[1], 'Sst-IRES-Cre')
    plot_cluster_proportion_cre(df,locations, fig, ax[0], 'Vip-IRES-Cre')
    plt.savefig(filedir + 'within_cluster_proportions.svg')
    plt.savefig(filedir + 'within_cluster_proportions.png')
    plt.savefig(filedir + 'within_cluster_proportions.pdf')


def compute_cluster_proportion_cre(df, cre, locations=None):
	'''
	Computes the proportion of cells in each cluster for each location (n_cells_in_cluster_for_loc / n_cells_in_loc)
	Then subtracts the average proportion of cells across locations (an estimate of overall cluster size?)

	'''
	if locations is None:
        # Get locations
    	locations = np.sort(df['location'].unique())
    else:
    	assert set(locations) == set(df['location'].unique()), "locations passed in don't match location column" 


    table = compute_proportion_cre(df, cre, locations)
>>>>>>> Stashed changes

    # get average proportion in each cluster
    table['mean'] = table.mean(axis=1) # average across locations (ex: if loc1 has 0.1% of cells and loc2 has 0.2, average is 0.15)

<<<<<<< Updated upstream
    # compute proportion in each area relative to cluster average
    for a in areas:
        table[a] = table[a] - table['mean']

    # plot proportions
    return table


def plot_cluster_proportion_cre(df, areas, fig, ax, cre='none', test='chi_squared_'):
=======
    # compute proportion in each location relative to average across locations
    for loc in locations:
        table[loc] = table[loc] - table['mean'] # proportion of cells in each location minus the average across locations?
    table = table[locations]

    return table


def plot_cluster_proportion_cre(df, locations, fig, ax, cre, test='chi_squared_'):
>>>>>>> Stashed changes
    '''
        Fraction of cells per location&depth 
    '''
<<<<<<< Updated upstream
    table = compute_cluster_proportion_cre(df, cre=cre, areas=areas)

    vmax = table.abs().max().max()
    vmax = .15
    # cmap = 'PRGn'
    cbar = ax.imshow(table, cmap='BrBG', vmin=-vmax, vmax=vmax)
    fig = fig.colorbar(cbar, ax=ax,  location='bottom', orientation='horizontal',
                       label='proportion of cells per location \n relative to cluster average')
    fig.ax.tick_params(labelsize=16)
    ax.set_xticks(range(0, len(areas)))
    ax.set_xticklabels(areas, rotation=90, size=16)
    ax.set_ylabel('Cluster ID', fontsize=20)
    ax.set_title(mapper(cre), fontsize=16)
=======
    table = compute_cluster_proportion_cre(df, cre, locations)

    vmax = table.abs().max().max()
    vmax = .15
    cbar = ax.imshow(table,cmap='PRGn',vmin=-vmax, vmax=vmax)
    fig.colorbar(cbar, ax=ax,label='proportion of cells per location \n relative to cluster average')
    ax.set_xticks(range(0,len(locations)))
    ax.set_xticklabels(locations,rotation=90)
    ax.set_ylabel('Cluster #',fontsize=16)  
    ax.set_title(mapper(cre),fontsize=16)
>>>>>>> Stashed changes
    ax.set_yticks(np.arange(len(table)))
    ax.set_yticklabels(np.arange(1, len(table)+1), size=16)

    # add statistics
<<<<<<< Updated upstream
    table2 = stats(df, cre=cre, areas=areas, test=test)
    for index in table2.index.values:
        if table2.loc[index]['bh_significant']:
            ax.plot(-1, index-1, 'r*', markersize=10)
    ax.set_xlim(-1.5, len(areas)-.5)
    ax.set_xticks(range(-1, len(areas)))
    ax.set_xticklabels(np.concatenate([['p<0.05'], areas]), rotation=90)
    ax.axvline(-0.5, color='k', linewidth=.5)
    plt.tight_layout()


def plot_cluster_percentages(df, areas=None):
    if areas is None:
        # Get areas
        areas = np.sort(df['location'].unique())
    else:
        assert set(areas) == set(df['location'].unique(
        )), "areas passed in don't match location column"
=======
    table2 = stats(df, cre,locations,test=test)
    for index in table2.index.values:
        if table2.loc[index]['bh_significant']:
            ax.plot(-1,index-1,'r*')
    ax.set_xlim(-1.5,len(locations)-.5)
    ax.set_xticks(range(-1,len(locations)))
    ax.set_xticklabels(np.concatenate([['p<0.05'],locations]),rotation=90)
    ax.axvline(-0.5,color='k',linewidth=.5)


def plot_cluster_percentages(df, locations=None):
    if locations is None:
        # Get locations
        locations = np.sort(df['location'].unique())
    else:
        assert set(locations) == set(df['location'].unique()), "locations passed in don't match location column" 
>>>>>>> Stashed changes

    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    fig.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.9, wspace=1)
<<<<<<< Updated upstream
    plot_cluster_percentage_cre(df, areas, fig, ax[2], 'Slc17a7-IRES2-Cre')
    plot_cluster_percentage_cre(df, areas, fig, ax[1], 'Sst-IRES-Cre')
    plot_cluster_percentage_cre(df, areas, fig, ax[0], 'Vip-IRES-Cre')
=======
    plot_cluster_percentage_cre(df,locations, fig, ax[2], 'Slc17a7-IRES2-Cre')
    plot_cluster_percentage_cre(df,locations, fig, ax[1], 'Sst-IRES-Cre')
    plot_cluster_percentage_cre(df,locations, fig, ax[0], 'Vip-IRES-Cre')
>>>>>>> Stashed changes
    plt.savefig(filedir+'within_cluster_percentages.svg')
    plt.savefig(filedir+'within_cluster_percentages.png')


<<<<<<< Updated upstream
def plot_cluster_percentage_cre(df, areas, fig, ax, cre='none', test='chi_squared_'):
=======
def plot_cluster_percentage_cre(df, locations, fig, ax, cre, test='chi_squared_'):
>>>>>>> Stashed changes
    '''
        Fraction of cells per location&depth 
    '''
<<<<<<< Updated upstream
    # count cells in each area/cluster
    if cre is 'none':
        table = df.groupby(['cluster_id', 'location'])[
            'cell_specimen_id'].count().unstack()
    else:
        table = df.query('cre_line == @cre').groupby(['cluster_id', 'location'])[
            'cell_specimen_id'].count().unstack()
    table = table[areas]
=======
    # count cells in each location/cluster
    table = df.query('cre_line == @cre').groupby(['cluster_id','location'])['cell_specimen_id'].count().unstack()
    table = table[locations]
>>>>>>> Stashed changes
    table = table.fillna(value=0)

    # compute proportion in each location/cluster
    for a in locations:
        table[a] = table[a]/table[a].sum()

    # get average proportion in each cluster
    table['mean'] = table.mean(axis=1)

<<<<<<< Updated upstream
    # build second table with cells in each area/cluster
    if cre is 'none':
        table2 = df.groupby(['cluster_id', 'location'])[
            'cell_specimen_id'].count().unstack()
    else:
        table2 = df.query('cre_line == @cre').groupby(['cluster_id', 'location'])[
            'cell_specimen_id'].count().unstack()
    table2 = table2[areas]
=======
    # build second table with cells in each location/cluster
    table2 = df.query('cre_line == @cre').groupby(['cluster_id','location'])['cell_specimen_id'].count().unstack()
    table2 = table2[locations]
>>>>>>> Stashed changes
    table2 = table2.fillna(value=0)

    # estimate chance cell counts based on average proportion in each cluster
    # then add relative fraction of actual counts compared to chance counts
    # subtract 1 so 0=chance
    for a in locations:
        table2[a+'_chance_count'] = table2[a].sum()*table['mean']
        table2[a+'_rel_fraction'] = table2[a]/table2[a+'_chance_count']-1
<<<<<<< Updated upstream

    area_rel_fraction = [area+'_rel_fraction' for area in areas]

    # plot proportions
    table2 = table2[area_rel_fraction]
    cbar = ax.imshow(table2, cmap='PRGn', vmin=-1, vmax=1)
    fig.colorbar(cbar, ax=ax, label='fraction of cells per area & depth')
    ax.set_ylabel('Cluster #', fontsize=16)
    ax.set_title(mapper(cre), fontsize=16)

    # add statistics
    table2 = stats(df, cre=cre, areas=areas, test=test)
    for index in table2.index.values:
        if table2.loc[index]['bh_significant']:
            ax.plot(-1, index-1, 'r*')
    ax.set_xlim(-1.5, len(areas)-.5)
    ax.set_xticks(range(-1, len(areas)))
    ax.set_xticklabels(np.concatenate([['p<0.05'], areas]), rotation=90)
    ax.axvline(-0.5, color='k', linewidth=.5)


def stats(df, cre='none', areas=[], test='chi_squared_', lambda_str='log-likelihood', cluster_range=range(0, 12)):
    '''
        Performs chi-squared tests to asses whether the observed cell counts 
        in each area/depth differ significantly from the average for that cluster. 
    '''

    # compute cell counts in each area/cluster
    if cre is 'none':
        table = df.groupby(['cluster_id', 'location'])[
            'cell_specimen_id'].count().unstack()
    else:
        table = df.query('cre_line == @cre').\
            groupby(['cluster_id', 'location'])[
            'cell_specimen_id'].count().unstack()
    table = table[areas]
=======
   
    location_rel_fraction = [location+'_rel_fraction' for location in locations]
 
    # plot proportions 
    table2 = table2[location_rel_fraction]
    cbar = ax.imshow(table2,cmap='PRGn',vmin=-1,vmax=1)
    fig.colorbar(cbar, ax=ax,label='fraction of cells per location & depth')
    ax.set_ylabel('Cluster #',fontsize=16)  
    ax.set_title(mapper(cre),fontsize=16)
    
    # add statistics
    table2 = stats(df, cre,locations,test=test)
    for index in table2.index.values:
        if table2.loc[index]['bh_significant']:
            ax.plot(-1,index-1,'r*')
    ax.set_xlim(-1.5,len(locations)-.5)
    ax.set_xticks(range(-1,len(locations)))
    ax.set_xticklabels(np.concatenate([['p<0.05'],locations]),rotation=90)
    ax.axvline(-0.5,color='k',linewidth=.5)


def stats(df, cre, locations=None, test='chi_squared_', lambda_str='log-likelihood'):
    '''
        Performs chi-squared tests to asses whether the observed cell counts in each location differ significantly 
        from the number of cells you would expect based on size of that cluster within that cre line

        chance = proportion_cells_in_cluster * total_n_cells_per_location
        actual = n_cells_in_cluster

        Assumes that input dataframe (usually cluster_meta table) contains a categorical column called 'location'
        the 'location' column is what will be used to group data to compute proportions per location per cluster

        lambda_str is for divergence test
    '''    
    if locations is None: 
       locations = np.sort(df['location'].unique())

    # compute cell counts in each location
    table = df.query('cre_line == @cre').\
        groupby(['cluster_id','location'])['cell_specimen_id'].count().unstack()
    table = table[locations]
>>>>>>> Stashed changes
    table = table.fillna(value=0)
    # this table has one column per location, with clusters as rows 

<<<<<<< Updated upstream
    # compute proportion for null hypothesis that areas have the same proportions
    table['total_cells'] = table.sum(axis=1)
    table['null_mean_proportion'] = table['total_cells'] / \
        np.sum(table['total_cells'])

    # second table of cell counts in each area/cluster
    if cre is 'none':
        table2 = df.groupby(['cluster_id', 'location'])[
            'cell_specimen_id'].count().unstack()
    else:
        table2 = df.query('cre_line == @cre').\
            groupby(['cluster_id', 'location'])[
            'cell_specimen_id'].count().unstack()
    table2 = table2[areas]
=======
    # compute proportion of cells per cluster for null hypothesis that locations have the same proportions
    table['total_cells'] = table.sum(axis=1) # total cells per cluster
    table['fraction_cells_per_cluster'] = table['total_cells']/np.sum(table['total_cells']) 
    # total cells per cluster divided by total cells across all locations and clusters (total for cre line)
    # fraction_cells_per_cluster (formerly called null_mean_proportion) is just the fraction of cells in each cluster within the cre line

    # second table of cell counts in each location
    table2 = df.query('cre_line == @cre').\
        groupby(['cluster_id','location'])['cell_specimen_id'].count().unstack()
    table2 = table2[locations]
>>>>>>> Stashed changes
    table2 = table2.fillna(value=0)
     # this table has one column per location, with clusters as rows 

    # compute estimated frequency of cells based on average fraction for each cluster
<<<<<<< Updated upstream
    for a in areas:
        table2[a+'_chance_count'] = table2[a].sum() * \
            table['null_mean_proportion']

    # perform chi-squared test
    area_chance = [area+'_chance_count' for area in areas]
    for index in table2.index.values:
        f = table2.loc[index][areas].values
        f_expected = table2.loc[index][area_chance].values
        not_f = table2[areas].sum().values - f

        # Manually doing check here bc Im on old version of scipy
        assert np.abs(np.sum(f) - np.sum(f_expected)) < 1, \
            'f and f_expected must be the same'

        if test == 'chi_squared_':
            out = chisquare(f, f_expected)
=======
    # chance count is the total number of cells in a given location times the average proportion of cells across locations
    # why not just take the total number of cells per cluster and divide by the number of locations?
    for loc in location:
    	total_cells_for_location = table2[loc].sum()
        table2[loc+'_chance_count'] = total_cells_for_location*table['fraction_cells_per_cluster']
    # chance count is the number of cells in a given location in a given cluster that you would find
    # if that cluster had x% of the total cells in that location, with x being the overall size of the cluster within the cre line

    # perform chi-squared test
    location_chance = [loc+'_chance_count' for loc in locations] # chance count is total cells per location * size of cluster (fractional)
    for index in table2.index.values: # iterate through cluster IDs
        f = table2.loc[index][locations].values # actual n_cells per cluster
        f_expected = table2.loc[index][location_chance].values # n cells per cluster based on size of cluster
        not_f = table2[locations].sum().values - f  # total number of cells in a given location minus n_cells per cluster for this location   
 		# not_f is for fischer test

        # Manually doing check here bc Im on old version of scipy
        # See this page for details on scipy versions & chi-square test assumptions: https://github.com/scipy/scipy/issues/14298
        assert np.abs(np.sum(f) - np.sum(f_expected))<1, \
            'f and f_expected must be the same'

        if test == 'chi_squared_':
            out = chisquare(f, f_expected) # is the actual number of cells in this location in this cluster different from expected number of cells based on cluster size?
>>>>>>> Stashed changes
            table2.at[index, test+'pvalue'] = out.pvalue
            table2.at[index, 'significant'] = out.pvalue < 0.05
        elif test == 'g_test_':
            f = f.astype(np.double)
            f_expected = f_expected.astype(np.double)
            out = power_divergence(f, f_expected, lambda_=lambda_str)
            table2.at[index, test+'pvalue'] = out.pvalue
            table2.at[index, 'significant'] = out.pvalue < 0.05
        elif test == 'fisher_':
<<<<<<< Updated upstream
            contingency = np.array([f, not_f])
=======
            contingency = np.array([f, not_f]) 
>>>>>>> Stashed changes
            if np.shape(contingency)[1] > 2:
                raise Exception(
                    'Need to import FisherExact package for non 2x2 tables')
                #pvalue = FisherExact.fisher_exact(contingency)
            else:
                oddsratio, pvalue = fisher_exact(contingency)
            table2.at[index, test+'pvalue'] = pvalue
            table2.at[index, 'significant'] = pvalue < 0.05

    # Use Benjamini Hochberg Correction for multiple comparisons
<<<<<<< Updated upstream
    table2 = add_hochberg_correction(table2, test=test)

    table2 = table2.reindex(cluster_range)
=======
    table2 = add_hochberg_correction(table2, test=test) 
>>>>>>> Stashed changes
    return table2


def mapper(cre):
    mapper = {
        'Slc17a7-IRES2-Cre': 'Excitatory',
        'Sst-IRES-Cre': 'Sst',
        'Vip-IRES-Cre': 'Vip',
        'none': 'All'}
    return mapper[cre]


def add_hochberg_correction(table, test='chi_squared_'):
    '''
        Performs the Benjamini Hochberg correction
    '''
    # Sort table by pvalues
    table = table.sort_values(by=test+'pvalue').reset_index()

    # compute the corrected pvalue based on the rank of each test
    # Need to use rank starting at 1
    table['imq'] = (1+table.index.values)/len(table)*0.05

    # Find the largest pvalue less than its corrected pvalue
    # all tests above that are significant
    table['bh_significant'] = False
    passing_tests = table[table[test+'pvalue'] < table['imq']]
    if len(passing_tests) > 0:
        last_index = table[table[test+'pvalue'] <
                           table['imq']].tail(1).index.values[0]
        table.at[last_index, 'bh_significant'] = True
        table.at[0:last_index, 'bh_significant'] = True

    # reset order of table and return
    return table.sort_values(by='cluster_id').set_index('cluster_id')
