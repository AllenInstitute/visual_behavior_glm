import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import visual_behavior.data_access.loading as loading
import visual_behavior.data_access.utilities as utilities

import visual_behavior_glm_strategy.GLM_fit_tools as gft
import visual_behavior_glm_strategy.GLM_params as glm_params
import visual_behavior_glm_strategy.GLM_visualization_tools as gvt

def load_across_session(run_params):
    glm_version = run_params['version']
    across_run_params = make_across_run_params(glm_version)
    filename = os.path.join(run_params['output_dir'],'across_df.pkl')
    across_df = pd.read_pickle(filename)
    return across_run_params, across_df

def make_across_run_params(glm_version):
    '''
        Makes a dummy dictionary with the figure directory hard coded
        This is only used as a quick fix for saving figures
    '''
    figdir = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/v_'\
        +glm_version+'/figures/across_session/'
    run_params = {}
    run_params['version'] = glm_version+'_across'
    run_params['figure_dir'] = figdir[:-1]
    run_params['include_4x2_data']=False
    return run_params

def plot_across_summary(across_df,across_run_params,savefig=False):
    '''
        Plots the population average dropout scores by experience and cre line,
        for the high level dropouts. Plots two versions, one with statistics 
        computed for the across session scores, the other for the within session 
        scores. 
    '''
    gvt.plot_population_averages(across_df, across_run_params, dropouts_to_show=[
        'all-images_within','omissions_within','behavioral_within','task_within'],
        across_session=True,stats_on_across=True,savefig=savefig)
    gvt.plot_population_averages(across_df, across_run_params, dropouts_to_show=[
        'all-images_within','omissions_within','behavioral_within','task_within'],
        across_session=True,stats_on_across=False,savefig=savefig)

def fraction_same(across_df):
    '''
        Prints a groupby table of the fraction of cells with coding scores
        that are the same between within and across normalization
    '''
    dropouts = ['omissions','all-images','behavioral','task']

    for dropout in dropouts:
        across_df[dropout+'_same'] = across_df[dropout+'_within'] == across_df[dropout+'_across']
    x = across_df.groupby(['cre_line','experience_level'])[['omissions_same','all-images_same',\
        'behavioral_same','task_same']].mean()
    print(x)
    return across_df

def scatter_df(across_df,cell_type, across_run_params,savefig=False):
    '''
        Plots a scatter plot comparing within and across coding scores
        for each of the high level dropouts for cells of <cell_type>
    '''   
 
    across_df = across_df.query('cell_type == @cell_type')

    fig, ax = plt.subplots(2,2,figsize=(11,8))
    plot_dropout(across_df, 'omissions', ax[0,0])
    plot_dropout(across_df, 'all-images', ax[0,1])
    plot_dropout(across_df, 'behavioral', ax[1,0])
    plot_dropout(across_df, 'task', ax[1,1])
    fig.suptitle(cell_type, fontsize=20)

    plt.tight_layout()
    if savefig:
        plt.savefig(across_run_params['figure_dir']+cell_type.replace(' ','_')+'_scatter.png')

def plot_dropout(across_df, dropout, ax):
    ''' 
        Helper function for scatter_df
    '''
    experience_levels = across_df['experience_level'].unique()
    colors = gvt.project_colors()
    for elevel in experience_levels:
        eacross_df = across_df.query('experience_level == @elevel')
        ax.plot(-eacross_df[dropout+'_within'],-eacross_df[dropout+'_across'],'o',\
            color=colors[elevel])
    ax.set_xlabel(dropout+' within',fontsize=18)
    ax.set_ylabel(dropout+' across',fontsize=18)
    ax.tick_params(axis='both',labelsize=16)

def get_cell_list(glm_version):
    run_params = glm_params.load_run_json(glm_version)
    include_4x2_data = run_params['include_4x2_data']
    cells_table = loading.get_cell_table(platform_paper_only=True,\
        include_4x2_data=include_4x2_data).reset_index()
    cells_table['passive'] = cells_table['passive'].astype(bool)
    cells_table = cells_table.query('not passive').copy()
    cells_table = utilities.limit_to_last_familiar_second_novel_active(cells_table)
    cells_table = utilities.limit_to_cell_specimen_ids_matched_in_all_experience_levels(cells_table)
    return cells_table

def load_cells(glm_version,clean_df=True): 
    '''
        Loads all cells that have across session coding scores computed.
        prints the cell_specimen_id for any cell that cannot be loaded.

        ARGS
        glm_version (str), name of glm version to use  
        clean_df (bool), return just the final dropout scores, or include the intermediate values
    
        RETURNS  
        df  - a dataframe containing the across and within session normalization
        fail_df - a dataframe containing cell_specimen_ids that could not be loaded    
    
    '''

    # 3921 unique cells
    print('Loading list of matched cells')
    cells_table = get_cell_list(glm_version)
    cells = cells_table['cell_specimen_id'].unique()

    dfs = []
    fail_to_load = []
    print('Loading across session normalized dropout scores')
    for cell in tqdm(cells):
        try:
            filename = '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/v_'\
                +glm_version+'/across_session/'+str(cell)+'.csv'
            score_df = pd.read_csv(filename)
            score_df['cell_specimen_id'] = cell
            if clean_df:
                columns = ['ophys_experiment_id','cell_specimen_id']+\
                    [x for x in score_df.columns if ('_within' in x) or ('_across' in x)]
                score_df = score_df[columns]
            else:
                score_df = score_df.drop(columns=['fit_index'],errors='ignore')
            dfs.append(score_df)
        except:
            fail_to_load.append(cell)
    print(str(len(fail_to_load))+' cells could not be loaded')

    # concatenate into one data frame, and merge in cell table data
    across_df = pd.concat(dfs) 
    across_df =  across_df.reset_index(drop=True)
    across_df['identifier'] = [str(x)+'_'+str(y) for (x,y) in zip(across_df['ophys_experiment_id'],across_df['cell_specimen_id'])]
    cells_table['identifier'] = [str(x)+'_'+str(y) for (x,y) in zip(cells_table['ophys_experiment_id'],cells_table['cell_specimen_id'])]
    across_df = pd.merge(across_df.drop(columns=['ophys_experiment_id','cell_specimen_id']), cells_table, on='identifier',suffixes=('','_y'),validate='one_to_one')

    # Assert that dropout scores are negative   
    kernels=['all-images','task','omissions','behavioral']
    for kernel in kernels:
        assert np.all(across_df[kernel+'_across']<=0), "Dropout scores must be negative"

    # Assert that across session dropout scores are less than equal to within cells
    # need to use a numerical tolerance because we divide by number of timesteps
    tol = 1e-8
    for kernel in kernels:
        assert np.all(across_df[kernel+'_within'] <= across_df[kernel+'_across']+tol), 'Across session dropouts must be less than or equal to within session dropouts'
 
    # Construct dataframe of cells that could not load, for debugging purposes
    if len(fail_to_load) > 0:
        fail_df = cells_table.query('cell_specimen_id in @fail_to_load')
        # The above command was behaving inconsistently. The code block
        # below is equivalent, including it for posterity. 
        # cells_table = cells_table.set_index('cell_specimen_id')
        # fail_df = cells_table.loc[fail_to_load]
    else:
        fail_df = pd.DataFrame()
        fail_df['cell_specimen_id'] = []

    # Assert that we have the correct number of cells
    assert len(across_df) + len(fail_df) == len(cells)*3, "incorrect number of cells"
    assert len(across_df['cell_specimen_id'].unique())+len(fail_df['cell_specimen_id'].unique()) == len(cells), "incorrect number of cells"

    return across_df, fail_df 

def compute_many_cells(cells,glm_version):
    ''' 
        For each cell_specimen_id in cells, compute the across session normalized dropout scores
        using the model in <glm_version>
    ''' 
    for cell in tqdm(cells):
        try:
            data, score_df = across_session_normalization(cell,glm_version)
        except:
            print(str(cell) +' crashed')
 
def across_session_normalization(cell_specimen_id, glm_version):
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
    include_4x2_data = run_params['include_4x2_data']
    cells_table = loading.get_cell_table(platform_paper_only=True, include_4x2_data=include_4x2_data)
    cells_table = cells_table.query('not passive').copy()
    cells_table = cells_table[cells_table['cell_specimen_id'] == cell_specimen_id]
    cells_table = cells_table.query('last_familiar_active or first_novel or second_novel_active')
    oeids = cells_table['ophys_experiment_id']

    # For each experiment, load the session, design matrix, and fit dictionary
    data = {}
    data['ophys_experiment_id'] = oeids
    print('Loading each experiment, will print a bunch of information about the design matrix for each experiment')
    glm_version = run_params['version']
    for oeid in oeids: 
        print('Loading oeid: '+str(oeid))
        run_params = glm_params.load_run_json(glm_version) 
        session, fit, design = gft.load_fit_experiment(oeid, run_params)       
        data[str(oeid)+'_session'] = session
        data[str(oeid)+'_fit'] = fit
        data[str(oeid)+'_design'] = design

    return data


def compute_across_session_dropouts(data, run_params, cell_specimen_id):
    '''
        Computes the across session dropout scores
        data                a dictionary containing the session object, fit dictionary, 
                            and design matrix for each experiment
        run_params          the paramater dictionary for this version
        cell_speciemn_id    the cell to compute the dropout scores for
    
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

        # Iterate over dropouts
        for dropout in dropouts:
            score_df.at[oeid, dropout] = results_df.loc[cell_specimen_id][dropout+'__avg_cv_adjvar_test']
            score_df.at[oeid, dropout+'_fc'] = results_df.loc[cell_specimen_id][dropout+'__avg_cv_adjvar_test_full_comparison']
            score_df.at[oeid, dropout+'_within'] = results_df.loc[cell_specimen_id][dropout+'__adj_dropout']

            # Get number of timestamps each kernel was active
            dropped_kernels = set(run_params['dropouts'][dropout]['dropped_kernels'])
            design_kernels = set(data[str(oeid)+'_design'].kernel_dict.keys())
            X = data[str(oeid)+'_design'].get_X(kernels=design_kernels.intersection(dropped_kernels))
            score_df.at[oeid, dropout+'_timestamps'] = np.sum(np.sum(np.abs(X.values),axis=1) > 0)

    # Iterate over dropouts and compute coding scores
    for dropout in dropouts:

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

    return score_df 
        
def print_df(score_df):
    '''
        Just useful for debugging
    '''
    dropouts = ['omissions','all-images','behavioral','task']
    for d in dropouts:
        #print(score_df[[d,d+'_within',d+'_across']])
        columns = [x for x in score_df.columns if d in x]
        print(score_df[columns])

def compare_across_df(across_df,dropout):
    '''
        Just for debugging
    '''
    plt.figure()
    plt.plot(np.sort(across_df[dropout+'_within'])[::-1],'bo-',label='within')
    plt.plot(np.sort(across_df[dropout+'_across'])[::-1],'rx-',label='across')
    plt.ylabel('dropout score')
    plt.xlabel('cell x session')
    plt.legend()

def append_kernel_excitation_across(weights_df, across_df):
    '''
        Appends labels about kernel weights from weights_df onto across_df 
        for some kernels, cells are labeled "excited" or "inhibited" if the 
        average weight over 750ms after the aligning event was positive 
        (excited), or negative (inhibited)

        Note that the excited/inhibited labels do not depend on within or   
        across session normalization since they are based on the weights 
        from the full model. 

        Additionally computes three coding scores for each kernel:
        kernel_across_positive is the across coding score if the kernel 
            was excited, otherwise 0
        kernel_across_negative is the across coding score if the kernel 
            was inhibited, otherwise 0
        kernel_across_signed is kernel_across_positive - kernel_across_negative

        across_df,_ = gas.load_cells()
        across_df = gas.append_kernel_excitation_across(weights_df, across_df) 
    '''   

    # Merge in three kernel metrics from weights_df 
    across_df = pd.merge(
        across_df,
        weights_df[['identifier','omissions_excited','all-images_excited','task_excited']],
        how = 'inner',
        on = 'identifier',
        validate='one_to_one'
        )
 
    # Use kernel metrics to define signed coding scores
    excited_kernels = ['omissions','task','all-images']
    for kernel in excited_kernels:
        across_df[kernel+'_across_positive'] = across_df[kernel+'_across']
        across_df[kernel+'_across_negative'] = across_df[kernel+'_across']
        across_df.loc[across_df[kernel+'_excited'] != True, kernel+'_across_positive'] = 0
        across_df.loc[across_df[kernel+'_excited'] != False,kernel+'_across_negative'] = 0   
        across_df[kernel+'_across_signed'] = across_df[kernel+'_across_positive'] - across_df[kernel+'_across_negative']
 
    for kernel in excited_kernels:
        assert np.all(across_df[kernel+'_across_positive']<=0), "Dropout scores must be negative"
        assert np.all(across_df[kernel+'_across_negative']<=0), "Dropout scores must be negative"
    return across_df


