import numpy as np
import pandas as pd
import visual_behavior_glm.GLM_fit_tools as gft
import visual_behavior_glm.GLM_params as glm_params
import visual_behavior_glm.GLM_visualization_tools as gvt
import visual_behavior.data_access.loading as loading
import visual_behavior.data_access.utilities as utilities
import matplotlib.pyplot as plt

figdir = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/v_24_events_all_L2_optimize_by_session/figures/across_session/'

def make_fake_run_params():
    '''
        Makes a dummy dictionary with the figure directory hard coded
    '''
    run_params = {}
    run_params['version'] = '24_events_all_L2_optimize_by_session_across'
    run_params['figure_dir'] = figdir[:-1]
    return run_params

def plot_across_summary(df):
    '''
        Plots the population average dropout scores by experience and cre line,
        for the high level dropouts. Plots two versions, one with statistics 
        computed for the across session scores, the other for the within session 
        scores. 
    '''
    run_params = make_fake_run_params()
    gvt.plot_population_averages(df, run_params, dropouts_to_show=[
        'all-images_within','omissions_within','behavioral_within','task_within'],across_session=True,stats_on_across=True)
    gvt.plot_population_averages(df, run_params, dropouts_to_show=[
        'all-images_within','omissions_within','behavioral_within','task_within'],across_session=True,stats_on_across=False)

def fraction_same(df):
    '''
        Prints a groupby table of the fraction of cells with coding scores
        that are the same between within and across normalization
    '''
    dropouts = ['omissions','all-images','behavioral','task']

    for dropout in dropouts:
        df[dropout+'_same'] = df[dropout+'_within'] == df[dropout+'_across']
    x = df.groupby(['cre_line','experience_level'])[['omissions_same','all-images_same','behavioral_same','task_same']].mean()
    print(x)
    return df

def scatter_df(df,cell_type):
    '''
        Plots a scatter plot comparing within and across coding scores
        for each of the high level dropouts
    '''   
 
    df = df.query('cell_type == @cell_type')

    fig, ax = plt.subplots(2,2,figsize=(11,8))
    plot_dropout(df, 'omissions', ax[0,0])
    plot_dropout(df, 'all-images', ax[0,1])
    plot_dropout(df, 'behavioral', ax[1,0])
    plot_dropout(df, 'task', ax[1,1])
    fig.suptitle(cell_type, fontsize=20)

    plt.tight_layout()
    plt.savefig(figdir+cell_type.replace(' ','_')+'_scatter.png')
    #plt.savefig(figdir+cell_type.replace(' ','_')+'_scatter.svg')

def plot_dropout(df, dropout, ax):
    experience_levels = df['experience_level'].unique()
    colors = gvt.project_colors()
    for elevel in experience_levels:
        edf = df.query('experience_level == @elevel')
        ax.plot(-edf[dropout+'_within'],-edf[dropout+'_across'],'o',color=colors[elevel])
    ax.set_xlabel(dropout+' within',fontsize=18)
    ax.set_ylabel(dropout+' across',fontsize=18)
    ax.tick_params(axis='both',labelsize=16)

def get_cell_list():
    '''
        Returns a list of cell_specimen_ids that are strictly matched
        and therefore used in the clustering analysis and across session normalization
    '''
    cells_table = loading.get_cell_table(platform_paper_only=True)
    cells_table = cells_table.query('not passive').copy()
    cells_table = utilities.limit_to_last_familiar_second_novel_active(cells_table)
    cells_table = utilities.limit_to_cell_specimen_ids_matched_in_all_experience_levels(cells_table)
    return cells_table
 

def load_cells(cells='all', glm_version ='24_events_all_L2_optimize_by_session'):
    '''
        Loads all cells that have across session coding scores computed.
        prints the cell_specimen_id for any cell that cannot be loaded.

        ARGS
        cells (str) if "examples" only returns the example cells
                    otherwise returns all cells.
    
        RETURNS  
        df  - a dataframe containing the across and within session normalization
        fail_to_load - a list of cell_specimen_ids that could not be loaded    
    
    '''
    if cells is 'examples':
        # list of cells marked as examples
        cells = [
            1086490397, 1086490480, 1086490510, 108649118, 
            1086559968, 1086559206, 1086551301, 1086490680, 
            1086490289, 1086490441]
    else:
        # 3921 unique cells
        cells = get_cell_list()['cell_specimen_id'].unique()

    dfs = []
    fail_to_load = []
    for cell in cells:
        try:
            filename = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/v_'+glm_version+'/across_session/'+str(cell)+'.csv' 
            score_df = pd.read_csv(filename)
            score_df['cell_specimen_id'] = cell
            #score_df = score_df.reset_index()
            dfs.append(score_df)
        except:
            print(str(cell)+' could not be loaded')
            fail_to_load.append(cell)
    df = pd.concat(dfs)
    df =  df.drop(columns = ['fit_index']).reset_index(drop=True)

    cells_table = loading.get_cell_table(platform_paper_only=True).reset_index()
    df['identifier'] = [str(x)+'_'+str(y) for (x,y) in zip(df['ophys_experiment_id'],df['cell_specimen_id'])]
    cells_table['identifier'] = [str(x)+'_'+str(y) for (x,y) in zip(cells_table['ophys_experiment_id'],cells_table['cell_specimen_id'])]
    df = pd.merge(df, cells_table, on='identifier',suffixes=('','_y'))
    
    return df, fail_to_load 

def compute_many_cells(cells):
    for cell in cells:
        try:
            data, score_df = across_session_normalization(cell)
        except:
            print(str(cell) +' crashed')
 
def across_session_normalization(cell_specimen_id =1086490680, glm_version='24_events_all_L2_optimize_by_session'):
    '''
        Computes the across session normalization for a cell
        This is very slow because we have to load the design matrices for each object
        Takes about 3 minutes. 
    
    '''
    run_params = glm_params.load_run_json(glm_version)
    data = get_across_session_data(run_params,cell_specimen_id)
    score_df = compute_across_session_dropouts(data, run_params, cell_specimen_id)
    filename = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/v_'+glm_version+'/across_session/'+str(cell_specimen_id)+'.csv'
    score_df.to_csv(filename)

    return data, score_df

def get_across_session_data(run_params, cell_specimen_id):
    '''
        Loads GLM information for each ophys experiment that this cell participated in.
        Very slow, takes about 3 minutes.
    '''

    # Find which experiments this cell was in
    cells_table = loading.get_cell_table(platform_paper_only=True)
    cells_table = cells_table.query('not passive').copy()
    cells_table = cells_table[cells_table['cell_specimen_id'] == cell_specimen_id]
    cells_table = cells_table.query('last_familiar_active or first_novel or second_novel_active')
    oeids = cells_table['ophys_experiment_id']

    # For each experiment, load the session, design matrix, and fit dictionary
    data = {}
    data['ophys_experiment_id'] = oeids
    print('Loading each experiment, will print a bunch of information about the design matrix for each experiment')
    for oeid in oeids: 
        print('Loading oeid: '+str(oeid))
        session, fit, design = gft.load_fit_experiment(oeid, run_params)       
        data[str(oeid)+'_session'] = session
        data[str(oeid)+'_fit'] = fit
        data[str(oeid)+'_design'] = design

    return data


def compute_across_session_dropouts(data, run_params, cell_specimen_id,clean_df = False):
    '''
        Computes the across session dropout scores
        data                a dictionary containing the session object, fit dictionary, 
                            and design matrix for each experiment
        run_params          the paramater dictionary for this version
        cell_speciemn_id    the cell to compute the dropout scores for
        clean_df            (bool) if True, returns only the within and across session dropout scores, 
                            otherwise returns intermediate values for error checking
    
    '''

    # Set up a dataframe to store across session coding scores
    df = pd.DataFrame()
    df['ophys_experiment_id'] =data['ophys_experiment_id']
    score_df = df.set_index('ophys_experiment_id')
    
    # Get list of dropouts to compute across session coding score
    dropouts = ['omissions','all-images','behavioral','task']

    # Iterate across three sessions to get VE of the dropout and full model
    for oeid in score_df.index.values:
        # Get the full comparison values and test values
        results_df = gft.build_dataframe_from_dropouts(data[str(oeid)+'_fit'], run_params)
        score_df['fit_index'] = np.where(data[str(oeid)+'_fit']['fit_trace_arr']['cell_specimen_id'].values == cell_specimen_id)[0][0]

        # Iterate over dropouts
        for dropout in dropouts:
            score_df.at[oeid, dropout] = results_df.loc[cell_specimen_id][dropout+'__avg_cv_adjvar_test']
            score_df.at[oeid,dropout+'_fc'] = results_df.loc[cell_specimen_id][dropout+'__avg_cv_adjvar_test_full_comparison']
            score_df.at[oeid, dropout+'_within'] = results_df.loc[cell_specimen_id][dropout+'__adj_dropout']

            # Get number of timestamps each kernel was active
            dropped_kernels = set(run_params['dropouts'][dropout]['dropped_kernels'])
            design_kernels = set(data[str(oeid)+'_design'].kernel_dict.keys())
            X = data[str(oeid)+'_design'].get_X(kernels=design_kernels.intersection(dropped_kernels))
            score_df.at[oeid, dropout+'_timestamps'] = np.sum(np.sum(np.abs(X.values),axis=1) > 0)

    # Iterate over dropouts and compute coding scores
    clean_columns = []
    for dropout in dropouts:
        clean_columns.append(dropout+'_within')
        clean_columns.append(dropout+'_across')

        # Adjust variance explained based on number of timestamps
        score_df[dropout+'_pt'] = score_df[dropout]/score_df[dropout+'_timestamps']   
        score_df[dropout+'_fc_pt'] = score_df[dropout+'_fc']/score_df[dropout+'_timestamps'] 

        # Determine which session had the highest variance explained    
        score_df[dropout+'_max'] = score_df[dropout+'_fc_pt'].max()

        # calculate across session coding scores
        score_df[dropout+'_across'] = -(score_df[dropout+'_fc_pt'] - score_df[dropout+'_pt'])/(score_df[dropout+'_max'])
        score_df.loc[score_df[dropout+'_across'] > 0,dropout+'_across'] = 0

        # Cleaning step for low VE dropouts
        score_df.loc[score_df[dropout+'_within'] == 0,dropout+'_across'] = 0

    # All done, cleanup
    if clean_df:
        score_df = score_df[clean_columns].copy()
    return score_df
        
def print_df(score_df):
    dropouts = ['omissions','all-images','behavioral','task']
    for d in dropouts:
        print(score_df[[d+'_within',d+'_across']])


