import numpy as np
import pandas as pd
from visual_behavior.data_access import reformat 
import visual_behavior_glm.GLM_fit_tools as gft
import visual_behavior_glm.build_dataframes as bd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
from tqdm import tqdm

def decode_experiment(oeid, data='events',window=[0,.75]):

    # Load SDK object
    print('Loading data')
    run_params = {'include_invalid_rois':False}
    session = gft.load_data(oeid, run_params)   
    reformat.add_licks_each_flash(session.stimulus_presentations, session.licks) 
    reformat.add_rewards_each_flash(session.stimulus_presentations, session.rewards)
    session.stimulus_presentations['licked'] = [True if len(licks) > 0 else False \
        for licks in session.stimulus_presentations.licks.values]
    session.stimulus_presentations['hit'] =  \
        session.stimulus_presentations['licked'] & \
        session.stimulus_presentations['is_change']
    session.stimulus_presentations['miss'] = \
        ~session.stimulus_presentations['licked'] & \
        session.stimulus_presentations['is_change']
    session.stimulus_presentations.at[~session.stimulus_presentations['is_change'],'hit'] = np.nan
    session.stimulus_presentations.at[~session.stimulus_presentations['is_change'],'miss'] = np.nan

    # get list of cell dataframes
    print('Generating cell dataframes')
    cells = get_cells(session,data=data,window=window)

    # Perform decoding iterating over the number of cells
    print('Decoding')
    results_df = iterate_n_cells(cells)

    # Save results
    print('Save results')
    filename='/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/decoding/experiments/' 
    filename += str(oeid)+'.pkl'
    print('Saving to: '+filename)
    results_df.to_pickle(filename)
    print('Finished')

def load_experiment_results(oeid):
    filename='/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/decoding/experiments/' 
    filename += str(oeid)+'.pkl'
    return pd.read_pickle(filename)

def load_all(experiment_table,summary_df):
    
    # Iterate through experiments and load decoding results
    dfs = []
    failed = 0
    for oeid in experiment_table.index.values:
        try:
            df = load_experiment_results(oeid)
        except:
            failed +=1
        else:
            df['ophys_experiment_id'] = oeid
            dfs.append(df)
    print('Failed to load: {}'.format(failed))
    
    # Merge and add experiment meta data
    df = pd.concat(dfs)
    df = pd.merge(df,experiment_table.reset_index()[['ophys_experiment_id',\
        'ophys_session_id','behavior_session_id','targeted_structure',\
        'imaging_depth','equipment_name']],on='ophys_experiment_id')
    df = pd.merge(df, summary_df[['ophys_session_id','visual_strategy_session','experience_level'\
        ,'cre_line']],on='ophys_session_id')

    return df 

def get_cells(session, data='events',window=[0,.75]):
    '''
        Iterate over all cells in this experiment and make a list
        of cell dataframes
    '''   
 
    # Iterate over all cells in experiment
    cells = []
    cell_specimen_ids = session.cell_specimen_table.index.values
    for cell in tqdm(cell_specimen_ids):
        # Generate the dataframe for this cell
        cell_df = get_cell_table(session, cell,data=data,window=window)
        cells.append(cell_df)
    
    # return list of cell dataframes
    return cells    
         

def get_cell_table(session, cell_specimen_id, data='events',window=[0,.75],balance=True):
    '''
        Generates a dataframe for one cell where rows are image presentations
        and the response at each timepoint is a column
    '''
    # Get cell activity interpolated onto constant time points
    df = bd.get_cell_df(session, cell_specimen_id, data=data)
    full_df = bd.get_cell_etr(df, session, time = window)
    
    # Pivot the table such that all responses at the same time point are a column
    cell = pd.pivot_table(full_df, values='response',
        index='stimulus_presentations_id', columns=['time'])
    
    # Annotate stimulus information
    cell = pd.merge(cell, session.stimulus_presentations, 
        left_index=True, right_index=True)

    # Balance classes by using just changes and the image before
    cell['pre_change'] = cell['is_change'].shift(-1,fill_value=False)
    if balance:
        cell = cell.query('is_change or pre_change')

    return cell

def get_matrix(cell):
    '''
        Grabs the responses of the cell at each time point across each image presentation
        and returns a numpy matrix 
    '''
    cols = np.sort([x for x in cell.columns.values if not isinstance(x,str)])
    x = cell[cols].to_numpy()
    return x

def plot_results_df(results_df):
    
    plt.figure()
    g = results_df.groupby('n_cells')
    summary = g.mean()
    summary['size'] = g.size()
    summary['sem_score'] = g['test_score'].sem()
    summary['sem_behavior_correlation'] = g['behavior_correlation'].sem()
    plt.errorbar(summary.index.values, summary['test_score'],yerr=summary['sem_score'],
        color='k')
    plt.errorbar(summary.index.values, summary['behavior_correlation'],
        yerr=summary['sem_behavior_correlation'],color='b')
    plt.plot(summary.index.values, summary.test_score,'ko-',label='test score')
    plt.plot(summary.index.values, summary.behavior_correlation,'bo-',
        label='beh. corr')
    plt.xlim(0,np.max(summary.index.values))
    plt.ylim(0,1)
    plt.ylabel('model performance',fontsize=16)
    plt.xlabel('number of cells',fontsize=16)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    plt.legend(loc='lower right')
    plt.tight_layout()

def iterate_n_cells(cells):
    n_cells = [1,2,5,10,20,40,80,160,320]
 
    results = {} 
    for n in n_cells:
        if len(cells) < n:
            break
        print('Decoding with n={} cells'.format(n))
        results[n] = decode_cells(cells, n)

    results_df = pd.concat(results)
    return results_df

def check_sampling():
    n_total = np.arange(30,500,10)
    def n_samples(n,k):
        return int(np.ceil(np.log(0.01)/np.log(1-k/n)))
    samples10 = [n_samples(x,10) for x in n_total]
    samples20 = [n_samples(x,20) for x in n_total]
    plt.figure()
    plt.plot(n_total, samples20,'ko-',label='20')
    plt.plot(n_total, samples10,'ro-',label='10')
    plt.ylabel('samples/experiment')
    plt.xlabel('cells in experiment')
    plt.figure()
    plt.plot(n_total, np.array(samples20)/np.array(n_total),'ko-',label='20')
    plt.plot(n_total, np.array(samples10)/np.array(n_total),'ro-',label='10')
    plt.xlabel('cells in experiment')
    plt.ylabel('samples/cell')
    plt.ylim(0,.6)

def decode_cells(cells, n_cells):
    '''
        Cells is a list of dataframes, one for each cell
        n_cells is the number of cells to decode with in each sample
    '''
    
    # How many times we need to sample in order to get 99% chance each cell 
    # is used at least once
    if n_cells == 1:
        n_samples = len(cells)
    else:
        n_samples = int(np.ceil(np.log(0.01)/np.log(1-n_cells/len(cells))))
    
    # Iterate over samples and save output
    output = []
    for n in tqdm(range(0,n_samples)):
        if n_cells == 1:
            temp = decode_cells_sample(cells, n_cells, index=n)
        else:
            temp = decode_cells_sample(cells, n_cells)
        output.append(temp)

    # Return the output of the samples
    output_df = pd.DataFrame(output)
    output_df['n_cells'] = n_cells
    return output_df

def decode_cells_sample(cells, n_cells,index=None):
    '''
        Sample from the list of cells, and perform decoding once
        returns the output of the decoding
    '''
 
    # Sample n_cells from list of cells
    if n_cells == 1:
        sample_cells = [cells[index]]
    else:
        cells_in_sample = np.random.choice(len(cells),n_cells, replace=False)
        sample_cells = [cells[i] for i in cells_in_sample] 
    
    # Construct X 
    X = []
    for cell in sample_cells:
        X.append(get_matrix(cell))
    X = np.concatenate(X,axis=1)
    
    # y is the same for every cell, since its a behavioral output
    y = sample_cells[0]['is_change'].values

    # run CV model
    model = {}
    rfc = RandomForestClassifier(class_weight='balanced')
    model['cv_prediction'] = cross_val_predict(rfc, X,y,cv=5)
    model['behavior_correlation'] = compute_behavior_correlation(sample_cells[0],model) 
    model['test_score'] = np.mean(y == model['cv_prediction'])

    # return results
    return model 
 
def compute_behavior_correlation(cell, model):
    cell['prediction'] = model['cv_prediction']   
    cell = cell.query('is_change')
    cell['hit'] = cell['hit'].astype(bool)
    return cell[['prediction','hit']].corr()['hit']['prediction']













     
