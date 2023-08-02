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
    for i in np.arange(100,1000,10):
        stats = compare_shuffle(n=i,num_shuffles=num_shuffles)
        pval = stats[0]
        chi = stats[1]
        if i==100:
            plt.plot(i,chi,'bo',label='chi-2')
            plt.plot(i,pval,'ro',label='distribution')
        else:
            plt.plot(i,chi,'bo')
            plt.plot(i,pval,'ro')

    plt.axhline(.05, linestyle='--',color='k')
    plt.legend()
    plt.ylabel('p value')
    plt.xlabel('num cells')
    plt.ylim(0,.5)

def compare_shuffle(n=100,p=.15,pn=.1,num_shuffles=1000):  
    # worried about independence
    num_h = int(np.floor(pn*n))
    num_m = n-num_h
    raw = [1]*(num_h)+[0]*(num_m)
    
    # Generate shuffle 
    num_hits =[]
    for i in np.arange(0,num_shuffles):
        #shuffle = np.random.rand(n) < pn
        shuffle = np.random.choice(raw,n)
        num_hits.append(np.sum(shuffle))

    # Compute chi-square where the data is 15%/85% of cells
    # and null is the mean % across shuffles
    data = [p*n*1000000000, (1-p)*n*1000000000]
    null = [np.mean(num_hits), n-np.mean(num_hits)]
    x = np.floor(np.array([data,null])).T
    out = chi2_contingency(x,correction=True)

    # Compare that with a p-value where we ask what percentage of the shuffles 
    # had more than 15%
    pval = np.sum(np.array(num_hits) >= p*n)/num_shuffles*2 

    return pval,out[1]


def final(df, cre,areas=None,test='chi_squared_'):
    '''
        Returns two tables
        proportion_table contains the proportion of cells in each location found in each cluster, relative to the average proportion across location for that cluster

        stats_table returns statistical tests on the proportion of cells in each location 
        Use 'bh_significant' unless you have a good reason to use the uncorrected tests

        areas should be a list of the locations to look at.
    '''
    if areas is None:
        # Get areas
        areas = np.sort(df['location'].unique())   
    else:
        assert set(areas) == set(df['location'].unique()), "areas passed in don't match location column" 

    proportion_table = compute_cluster_proportion_cre(df, cre,areas)
    stats_table = stats(df,cre,areas,test=test)
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
    plot_cluster_percentages(df)   
 
def load_cluster_labels():
    '''
        - Loads a dataframe of cluster labels
        - merges in cell table data 
        - defines a `location` column with depth/area combinations
        - drops clusters with less than 5 cells
    '''

    # Load cluster labels
    filepath = '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/platform_paper_plots/figure_4/24_events_all_L2_optimize_by_session/220622_across_session_norm_10_5_10/cluster_labels_Vip_10_Sst_5_Slc17a7_10.h5'
    df = pd.read_hdf(filepath, key='df')

    # load cell data
    cells_table = loading.get_cell_table(platform_paper_only=True,include_4x2_data=False).reset_index().drop_duplicates(subset='cell_specimen_id')
    df = df.drop(columns=['labels','cre_line'])
    df = pd.merge(df, cells_table, on='cell_specimen_id',suffixes=('','_y'))  

    # Bin depths and annotate
    df['coarse_binned_depth'] = ['upper' if x < 250 else 'lower' for x in df['imaging_depth']]
    df['location'] = df['targeted_structure']+'_'+df['coarse_binned_depth']

    # Remove clusters with less than 5 cells
    #df = df.drop(df.index[(df['cre_line']=="Sst-IRES-Cre")&(df['cluster_id']==6)])
    #df = df.drop(df.index[(df['cre_line']=="Slc17a7-IRES2-Cre")&(df['cluster_id']==10)])

    return df

def plot_proportions(df,areas=None,savefig=False,extra='',test='chi_squared_'):
    '''
        Compute, then plot, the proportion of cells in each location within each cluster
    '''
    
    if areas is None:
        # Get areas
        areas = np.sort(df['location'].unique())   
    else:
        assert set(areas) == set(df['location'].unique()), "areas passed in don't match location column" 

    fig, ax = plt.subplots(1,3,figsize=(8,4))
    fig.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.9, wspace=1)
    plot_proportion_cre(df,areas, fig, ax[2], 'Slc17a7-IRES2-Cre',test=test)
    plot_proportion_cre(df,areas,  fig, ax[1], 'Sst-IRES-Cre',test=test)
    plot_proportion_cre(df,areas,  fig, ax[0], 'Vip-IRES-Cre',test=test)
    if savefig:
        extra = extra+'_'+test
        plt.savefig(filedir+'cluster_proportions'+extra+'.svg')
        plt.savefig(filedir+'cluster_proportions'+extra+'.png')

def compute_proportion_cre(df, cre,areas):
    '''
        Computes the proportion of cells in each cluster within each location
        
        location must be a column of string names. The statistics are computed relative to whatever the location column contains
    '''

    # Count cells in each area/cluster
    table = df.query('cre_line == @cre').groupby(['cluster_id','location'])['cell_specimen_id'].count().unstack()
    table = table[areas]
    table = table.fillna(value=0)

    # compute fraction in each area/cluster
    for a in areas:
        table[a] = table[a]/table[a].sum()
    return table

def plot_proportion_cre(df,areas, fig,ax, cre,test='chi_squared_'):
    '''
        Fraction of cells per area&depth 
    '''

    # Get proportions
    table = compute_proportion_cre(df, cre,areas)

    # plot proportions
    cbar = ax.imshow(table,cmap='Purples',vmax=.4)
    ax.set_ylabel('Cluster #',fontsize=16)  
    ax.set_yticks(range(0,len(table)))
    ax.set_yticklabels(range(1,len(table)+1))
    ax.set_title(mapper(cre),fontsize=16) 
    fig.colorbar(cbar, ax=ax,label='fraction of cells per location')
   
    # Add statistics 
    table2 = stats(df, cre,areas,test=test)
    for index in table2.index.values:
        if table2.loc[index]['bh_significant']:
            ax.plot(-1,index-1,'r*')

    num_areas = len(areas)
    ax.set_xlim(-1.5,num_areas-.5)
    ax.set_xticks(range(-1,num_areas))
    ax.set_xticklabels(np.concatenate([[test[:-1]],areas]),rotation=90)
    ax.axvline(-0.5,color='k',linewidth=.5)

def plot_proportion_differences(df,areas=None):
    '''
        Computes, then plots, the proportion of cells in each location within each cluster
        relative to a 1/n average distribution across n clusters. 
    '''

    if areas is None:
        # Get areas
        areas = np.sort(df['location'].unique())    
    else:
        assert set(areas) == set(df['location'].unique()), "areas passed in don't match location column" 

    fig, ax = plt.subplots(1,3,figsize=(8,4))
    fig.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.9, wspace=1)
    plot_proportion_differences_cre(df, areas, fig, ax[2], 'Slc17a7-IRES2-Cre')
    plot_proportion_differences_cre(df, areas, fig, ax[1], 'Sst-IRES-Cre')
    plot_proportion_differences_cre(df, areas, fig, ax[0], 'Vip-IRES-Cre')
    plt.savefig(filedir+'cluster_proportion_differences.svg')
    plt.savefig(filedir+'cluster_proportion_differences.png')

def compute_proportion_differences_cre(df, cre,areas):
    '''
        compute proportion differences relative to 1/n average
    '''
    # count cells in each area/cluster
    table = df.query('cre_line == @cre').groupby(['cluster_id','location'])['cell_specimen_id'].count().unstack()
    table = table[areas]
    table = table.fillna(value=0)
    nclusters = len(table.index.values)

    # compute fraction in each area relative to expected fraction 
    for a in areas:
        table[a] = table[a]/table[a].sum() - 1/nclusters
    return table

def plot_proportion_differences_cre(df,areas, fig,ax, cre):
    '''
        Fraction of cells per location, then
        subtract expected fraction (1/n)
    '''
    table = compute_proportion_differences_cre(df,cre,areas)

    # plot fractions
    vmax = table.abs().max().max()
    cbar = ax.imshow(table,cmap='PRGn',vmin=-vmax,vmax=vmax)
    fig.colorbar(cbar, ax=ax,label='fraction of cells per location \nrelative to evenly distributed across clusters')
    ax.set_xticks(range(0,len(areas)))
    ax.set_xticklabels(areas,rotation=90)
    ax.set_ylabel('Cluster #',fontsize=16)  
    ax.set_title(mapper(cre),fontsize=16) 

def plot_cluster_proportions(df,areas=None):
    if areas is None:
        # Get areas
        areas = np.sort(df['location'].unique())    
    else:
        assert set(areas) == set(df['location'].unique()), "areas passed in don't match location column" 

    fig, ax = plt.subplots(1,3,figsize=(14,8))
    fig.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.9, wspace=1)
    plot_cluster_proportion_cre(df,areas, fig, ax[2], 'Slc17a7-IRES2-Cre')
    plot_cluster_proportion_cre(df,areas, fig, ax[1], 'Sst-IRES-Cre')
    plot_cluster_proportion_cre(df,areas, fig, ax[0], 'Vip-IRES-Cre')
    plt.savefig(filedir+'within_cluster_proportions.svg')
    plt.savefig(filedir+'within_cluster_proportions.png')
    plt.savefig(filedir + 'within_cluster_proportions.pdf')

def compute_cluster_proportion_cre(df, cre,areas):
    table = compute_proportion_cre(df, cre,areas)

    # get average proportion in each cluster
    table['mean'] = table.mean(axis=1)

    # compute proportion in each area relative to cluster average
    for a in areas:
        table[a] = table[a] - table['mean'] 

    # plot proportions
    table = table[areas]
    return table

def plot_cluster_proportion_cre(df,areas,fig,ax, cre,test='chi_squared_'):
    '''
        Fraction of cells per area&depth 
    '''
    table = compute_cluster_proportion_cre(df,cre,areas)

    vmax = table.abs().max().max()
    vmax = .15
    cbar = ax.imshow(table,cmap='PRGn',vmin=-vmax, vmax=vmax)
    fig.colorbar(cbar, ax=ax,label='proportion of cells per location \n relative to cluster average')
    ax.set_xticks(range(0,len(areas)))
    ax.set_xticklabels(areas,rotation=90)
    ax.set_ylabel('Cluster #',fontsize=16)  
    ax.set_title(mapper(cre),fontsize=16)
    ax.set_yticks(np.arange(len(table)))
    ax.set_yticklabels(np.arange(1,len(table)+1))
    
    # add statistics
    table2 = stats(df, cre,areas,test=test)
    for index in table2.index.values:
        if table2.loc[index]['bh_significant']:
            ax.plot(-1,index-1,'r*')
    ax.set_xlim(-1.5,len(areas)-.5)
    ax.set_xticks(range(-1,len(areas)))
    ax.set_xticklabels(np.concatenate([['p<0.05'],areas]),rotation=90)
    ax.axvline(-0.5,color='k',linewidth=.5)

def plot_cluster_percentages(df,areas=None):
    if areas is None:
        # Get areas
        areas = np.sort(df['location'].unique())    
    else:
        assert set(areas) == set(df['location'].unique()), "areas passed in don't match location column" 

    fig, ax = plt.subplots(1,3,figsize=(8,4))
    fig.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.9, wspace=1)
    plot_cluster_percentage_cre(df,areas, fig, ax[2], 'Slc17a7-IRES2-Cre')
    plot_cluster_percentage_cre(df,areas, fig, ax[1], 'Sst-IRES-Cre')
    plot_cluster_percentage_cre(df,areas, fig, ax[0], 'Vip-IRES-Cre')
    plt.savefig(filedir+'within_cluster_percentages.svg')
    plt.savefig(filedir+'within_cluster_percentages.png')

def plot_cluster_percentage_cre(df,areas,fig,ax, cre,test='chi_squared_'):
    '''
        Fraction of cells per area&depth 
    '''
    # count cells in each area/cluster
    table = df.query('cre_line == @cre').groupby(['cluster_id','location'])['cell_specimen_id'].count().unstack()
    table = table[areas]
    table = table.fillna(value=0)

    # compute proportion in each area/cluster
    for a in areas:
        table[a] = table[a]/table[a].sum()

    # get average proportion in each cluster
    table['mean'] = table.mean(axis=1)

    # build second table with cells in each area/cluster
    table2 = df.query('cre_line == @cre').groupby(['cluster_id','location'])['cell_specimen_id'].count().unstack()
    table2 = table2[areas]
    table2 = table2.fillna(value=0)

    # estimate chance cell counts based on average proportion in each cluster
    # then add relative fraction of actual counts compared to chance counts
    # subtract 1 so 0=chance
    for a in areas:
        table2[a+'_chance_count'] = table2[a].sum()*table['mean']
        table2[a+'_rel_fraction'] = table2[a]/table2[a+'_chance_count']-1
   
    area_rel_fraction = [area+'_rel_fraction' for area in areas]
 
    # plot proportions 
    table2 = table2[area_rel_fraction]
    cbar = ax.imshow(table2,cmap='PRGn',vmin=-1,vmax=1)
    fig.colorbar(cbar, ax=ax,label='fraction of cells per area & depth')
    ax.set_ylabel('Cluster #',fontsize=16)  
    ax.set_title(mapper(cre),fontsize=16)
    
    # add statistics
    table2 = stats(df, cre,areas,test=test)
    for index in table2.index.values:
        if table2.loc[index]['bh_significant']:
            ax.plot(-1,index-1,'r*')
    ax.set_xlim(-1.5,len(areas)-.5)
    ax.set_xticks(range(-1,len(areas)))
    ax.set_xticklabels(np.concatenate([['p<0.05'],areas]),rotation=90)
    ax.axvline(-0.5,color='k',linewidth=.5)

def stats(df,cre,areas,test='chi_squared_',lambda_str='log-likelihood'):
    '''
        Performs chi-squared tests to asses whether the observed cell counts 
        in each area/depth differ significantly from the average for that cluster. 
    '''    

    # compute cell counts in each area/cluster
    table = df.query('cre_line == @cre').\
        groupby(['cluster_id','location'])['cell_specimen_id'].count().unstack()
    table = table[areas]
    table = table.fillna(value=0)

    # compute proportion for null hypothesis that areas have the same proportions
    table['total_cells'] = table.sum(axis=1)
    table['null_mean_proportion'] = table['total_cells']/np.sum(table['total_cells'])

    # second table of cell counts in each area/cluster
    table2 = df.query('cre_line == @cre').\
        groupby(['cluster_id','location'])['cell_specimen_id'].count().unstack()
    table2 = table2[areas]
    table2 = table2.fillna(value=0)

    # compute estimated frequency of cells based on average fraction for each cluster
    for a in areas:
        table2[a+'_chance_count'] = table2[a].sum()*table['null_mean_proportion']

    # perform chi-squared test
    area_chance = [area+'_chance_count' for area in areas]
    for index in table2.index.values:
        f = table2.loc[index][areas].values
        f_expected = table2.loc[index][area_chance].values
        not_f = table2[areas].sum().values - f       
 
        # Manually doing check here bc Im on old version of scipy
        assert np.abs(np.sum(f) - np.sum(f_expected))<1, \
            'f and f_expected must be the same'

        if test == 'chi_squared_':
            out = chisquare(f,f_expected)
            table2.at[index, test+'pvalue'] = out.pvalue
            table2.at[index, 'significant'] = out.pvalue < 0.05
        elif test == 'g_test_':
            f = f.astype(np.double)
            f_expected = f_expected.astype(np.double)
            out = power_divergence(f, f_expected,lambda_=lambda_str)
            table2.at[index, test+'pvalue'] = out.pvalue
            table2.at[index, 'significant'] = out.pvalue < 0.05           
        elif test == 'fisher_':
            contingency = np.array([f,not_f]) 
            if np.shape(contingency)[1] > 2:
                raise Exception('Need to import FisherExact package for non 2x2 tables')
                #pvalue = FisherExact.fisher_exact(contingency)
            else:
                oddsratio, pvalue = fisher_exact(contingency)
            table2.at[index, test+'pvalue'] = pvalue
            table2.at[index, 'significant'] = pvalue < 0.05              

    # Use Benjamini Hochberg Correction for multiple comparisons
    table2 = add_hochberg_correction(table2,test=test) 
    return table2

def mapper(cre):
    mapper = {
        'Slc17a7-IRES2-Cre':'Excitatory',
        'Sst-IRES-Cre':'Sst',
        'Vip-IRES-Cre':'Vip',
        }
    return mapper[cre]

def add_hochberg_correction(table,test='chi_squared_'):
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
    if len(passing_tests) >0:
        last_index = table[table[test+'pvalue'] < table['imq']].tail(1).index.values[0]
        table.at[last_index,'bh_significant'] = True
        table.at[0:last_index,'bh_significant'] = True
    
    # reset order of table and return
    return table.sort_values(by='cluster_id').set_index('cluster_id') 
 

