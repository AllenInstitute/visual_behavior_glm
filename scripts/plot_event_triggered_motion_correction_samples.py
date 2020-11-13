from visual_behavior.data_access import loading
import visual_behavior_glm.GLM_analysis_tools as gat
import visual_behavior_glm.GLM_visualization_tools as gvt
from visual_behavior_glm.glm import GLM
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import visual_behavior.utilities as vbu
import seaborn as sns
import visual_behavior.plotting as vbp
import os

savepath = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/lick-triggered_avg_plots'

glm_version='9a_L2_optimize_by_session'
dropout_summary = gat.retrieve_results({'glm_version':glm_version}, results_type='summary')

lick_cells = dropout_summary.query('dropout=="single-licks" and adj_fraction_change_from_full < -0.2')

for idx,lick_cell in lick_cells.sample(100).iterrows():
    dropout='single-licks'
    ophys_experiment_id = lick_cell['ophys_experiment_id']
    cell_specimen_id = lick_cell['cell_specimen_id']

    res = gat.retrieve_results({
        'glm_version':glm_version,
        'ophys_experiment_id':int(ophys_experiment_id),
        'cell_specimen_id':int(cell_specimen_id),
        'dropout':dropout
    }, results_type='summary').iloc[0]

    title='Experiment ID {}\nCell Specimen ID {}\n{} adj_fraction_change_from_full = {:0.4f}'.format(
        res['ophys_experiment_id'],
        res['cell_specimen_id'],
        res['dropout'],
        res['adj_fraction_change_from_full']
    )
    print('making plot for\n {}'.format(title))
    fig,ax = gvt.plot_lick_triggered_motion(lick_cell['ophys_experiment_id'], lick_cell['cell_specimen_id'], title=title)

    vbp.save_figure(fig, fname=os.path.join(savepath, '{}.png'.format(title)), size=(15,8))