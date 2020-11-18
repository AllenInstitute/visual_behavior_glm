import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import visual_behavior_glm.GLM_params as glm_params
import visual_behavior_glm.GLM_visualization_tools as gvt
from visual_behavior_glm.glm import GLM
plt.ion()


if False: # Code snippets for doing basic analyses. 
    # Make run JSON
    VERSION = 1
    src_path = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/GLM/visual_behavior_glm/' 
    glm_params.make_run_json(VERSION,label='testing',username='alex', src_path = src_path, TESTING=True)
   
    # Import code from a specific version
    # import_dir = self.run_params['model_freeze_dir'].rstrip('/')
    # module_name = 'GLM_fit_tools'
    # file_path = os.path.join(import_dir, module_name+'.py')
    # print('importing {} from {}'.format(module_name, file_path))

    # spec = importlib.util.spec_from_file_location(module_name, file_path)
    # gft = importlib.util.module_from_spec(spec)
    # sys.modules[module_name] = gft
    # spec.loader.exec_module(gft)

    # To run just one session:
    run_params = glm_params.load_run_json('4_L2_optimize_by_cell')
    oeid = run_params['ophys_experiment_ids'][-1]

    # Load existing results
    session, fit, design = gft.load_fit_experiment(oeid, run_params)

    # Fit results
    session, fit, design = gft.fit_experiment(oeid, run_params)

    # Analyze drop results
    drop_results = gft.build_dataframe_from_dropouts(fit)
    L2_results = gft.L2_report(fit)
    
    # Make GLM object
    g = glm.GLM(oeid, VERSION, use_previous_fit=True, log_results=False, log_weights=False)
    

    # Analysis figures 
    # results summary # results are experiment/cell/dropout
    results = gat.retrieve_results(search_dict={'glm_version':version}, results_type='summary')
 
    # results_pivoted # rows are experiment/cell
    results_pivoted = gat.build_pivoted_results_summary('adj_fraction_change_from_full',results_summary=results)
 
    # weights_df
    weights_df = gat.build_weights_df(run_params, results_pivoted)
    gvt.all_kernels_evaluation(weights_df,run_params)
    gvt.all_kernels_evaluation(weights_df,run_params,equipment_filter="mesoscope")
    gvt.all_kernels_evaluation(weights_df,run_params,equipment_filter="scientifica")
    gvt.all_kernels_evaluation(weights_df,run_params,session_filter=[1])
    gvt.all_kernels_evaluation(weights_df,run_params,session_filter=[2])
    gvt.all_kernels_evaluation(weights_df,run_params,session_filter=[3])
    gvt.all_kernels_evaluation(weights_df,run_params,session_filter=[4])
    gvt.all_kernels_evaluation(weights_df,run_params,session_filter=[5])
    gvt.all_kernels_evaluation(weights_df,run_params,session_filter=[6])
    gvt.all_kernels_evaluation(weights_df,run_params,depth_filter=[0,299])
    gvt.all_kernels_evaluation(weights_df,run_params,depth_filter=[299,1000])



    # Make over-fitting figures
    # Results_full
    full_results = gat.retrieve_results(search_dict={'glm_version':version}, results_type='full')
    gat.compute_over_fitting_proportion(full_results, run_params) 
    gvt.plot_over_fitting_summary(full_results, run_params)
    gvt.plot_all_over_fitting(full_results, run_params)

def setup_schematic():
    version = '8a_L2_optimize_by_session'
    oeid = 830700781
    glm = GLM(oeid,version=version, log_results=False, log_weights=False, use_previous_fit=True)
    return glm 

def make_schematic(glm,cell=1028768972,t_range=5,time_to_plot=3291,alpha=.25):
    t_span = (time_to_plot-t_range, time_to_plot+t_range)
    fig, ax = gvt.make_cosyne_summary_figure(glm, cell, t_span,alpha=alpha)
    ax['visual_kernels'].set_ylabel('Kernel Output',fontsize=14)
    ax['visual_kernels'].set_xlabel('Time (s)',fontsize=14)
    ax['behavioral_kernels'].set_ylabel('Kernel Output',fontsize=14)
    ax['behavioral_kernels'].set_xlabel('Time (s)',fontsize=14)
    ax['cognitive_kernels'].set_ylabel('Kernel Output',fontsize=14)
    ax['cognitive_kernels'].set_xlabel('Time (s)',fontsize=14)
    ax['visual_kernels'].set_xlim(t_span) 
    ax['behavioral_kernels'].set_xlim(t_span) 
    ax['cognitive_kernels'].set_xlim(t_span)
    ax['cell_response'].set_xlim(t_span)
    ax['cell_response'].set_ylabel('$\Delta$ F/F',fontsize=14)
    ax['cell_response'].set_xlabel('Time (s)',fontsize=14)
    ax['cell_response'].tick_params(axis='both',labelsize=12)
    ax['visual_kernels'].tick_params(axis='both',labelsize=12) 
    ax['behavioral_kernels'].tick_params(axis='both',labelsize=12) 
    ax['cognitive_kernels'].tick_params(axis='both',labelsize=12)
    ax['visual_kernels'].axhline(0,color='k',alpha=.25) 
    ax['behavioral_kernels'].axhline(0,color='k',alpha=.25) 
    ax['cognitive_kernels'].axhline(0,color='k',alpha=.25)
    ax['cell_response'].axhline(0,color='k',alpha=.25)
    ax['cell_response'].set_ylim(list(np.array(ax['cell_response'].get_ylim())*1.1))
    return fig, ax


def make_dropout_summary_plot(dropout_summary):
    fig, ax = plt.subplots(figsize=(10,8))
    dropouts_to_show = ['visual','behavioral','cognitive']
    gvt.plot_dropout_summary_cosyne(dropout_summary, ax, dropouts_to_show)
    ax.tick_params(axis='both',labelsize=20)
    ax.set_ylabel('% decrease in variance explained \n when removing sets of kernels',fontsize=24)
    ax.set_xlabel('Sets of Kernels',fontsize=24)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(['Visual','Behavioral','Cognitive'])
    ax.axhline(0, color='k',linestyle='--',alpha=.25)
    y = ax.get_yticks()
    ax.set_yticklabels(np.round(y*100).astype(int))
    plt.tight_layout()
    return ax
