import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import visual_behavior_glm.GLM_params as glm_params
import visual_behavior_glm.GLM_visualization_tools as gvt
import visual_behavior_glm.GLM_analysis_tools as gat
from visual_behavior_glm.glm import GLM

if False: # Code snippets for doing basic analyses. 

    # Make run JSON
    #####################
    VERSION = 1
    src_path = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/GLM/visual_behavior_glm/' 
    glm_params.make_run_json(VERSION,label='testing',username='alex', src_path = src_path, TESTING=True)
   


    # Import code from a specific version
    #####################
    import_dir = self.run_params['model_freeze_dir'].rstrip('/')
    module_name = 'GLM_fit_tools'
    file_path = os.path.join(import_dir, module_name+'.py')
    print('importing {} from {}'.format(module_name, file_path))

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    gft = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = gft
    spec.loader.exec_module(gft)



    # Fit Model
    #####################

    # To run just one session:
    run_params = glm_params.load_run_json('4_L2_optimize_by_cell')
    oeid = run_params['ophys_experiment_ids'][-1]

    # Fit results
    session, fit, design = gft.fit_experiment(oeid, run_params)



    # Load basic results
    #####################

    # Load existing results
    session, fit, design = gft.load_fit_experiment(oeid, run_params)

    # Analyze drop results
    drop_results = gft.build_dataframe_from_dropouts(fit)
    L2_results = gft.L2_report(fit)
    
    # Make GLM object
    g = glm.GLM(oeid, VERSION, use_previous_fit=True, log_results=False, log_weights=False)
    


    # Analysis Dataframes 
    #####################

    # results summary # results are experiment/cell/dropout
    results = gat.retrieve_results(search_dict={'glm_version':version}, results_type='summary')
 
    # results_pivoted # rows are experiment/cell
    results_pivoted = gat.build_pivoted_results_summary('adj_fraction_change_from_full',results_summary=results)

    # Full Results
    full_results = gat.retrieve_results(search_dict={'glm_version':version}, results_type='full')
 
    # weights_df
    weights_df = gat.build_weights_df(run_params, results_pivoted)



    # Analysis Figures
    #####################

    # Make Nested Model plot (rainbow plot)
    gvt.plot_dropouts(run_params)

    # Make plot of kernel support
    gvt.plot_kernel_support(g)

    # Make over-fitting figures
    # You may need to `mkdir over_fitting_figures` 
    gat.compute_over_fitting_proportion(full_results, run_params) 
    gvt.plot_over_fitting_summary(full_results, run_params)
    gvt.plot_all_over_fitting(full_results, run_params)

    # Make Coding Fraction plots
    # You may need to `mkdir coding` 
    gvt.plot_coding_fraction(results_pivoted, 'omissions') # Example
    gvt.plot_all_coding_fraction(results_pivoted, run_params, metric='fraction') # Make them all

    # Make Kernel figures
    # You may need to `mkdir kernels` 
    gvt.kernel_evaluation(weights_df, run_params, 'omissions') # Example
    gvt.all_kernels_evaluation(weights_df,run_params) # Make them all
    
    # Make Kernel Comparison Figures
    gvt.plot_kernel_comparison(weights_df, run_params, 'omissions',cell_filter='vip',compare=['session'],plot_errors=False) # Example
    gvt.plot_all_kernel_comparison(weights_df, run_params, cell_filter='vip',compare=['session'],plot_errors=False) # Make them all
    
    ## Comparing Versions
    #####################
    versions = ['9a_L2_optimize_by_session','10a_L2_optimize_by_session']
    comparison_table = gat.get_glm_version_comparison_table(versions)
    jointplot = gvt.plot_glm_version_comparison(comparison_table, versions_to_compare=versions)

    ## 3D plots
    #############
    all_traces = gvt.plot_kernel_comparison(weights_df, run_params, 'omissions')
    



def get_analysis_dfs(VERSION):
    run_params = glm_params.load_run_json(VERSION)
    results = gat.retrieve_results(search_dict={'glm_version':VERSION}, results_type='summary')
    results_pivoted = gat.build_pivoted_results_summary('adj_fraction_change_from_full',results_summary=results)
    full_results = gat.retrieve_results(search_dict={'glm_version':VERSION}, results_type='full')
    weights_df = gat.build_weights_df(run_params, results_pivoted)  
    add_categorical(weights_df) 
    add_categorical(results_pivoted) 
    return run_params, results, results_pivoted, weights_df, full_results

def add_categorical(df):
    '''
        Adds categorical string labels, useful for plotting
    '''
    df['session'] = [str(x) for x in df['session_number']]
    df['active'] = ['active' if x in [1,3,4,6] else 'passive' for x in df['session_number']]
    df['familiar'] = ['familiar' if x in [1,2,3] else 'novel' for x in df['session_number']]
    df['equipment'] =['scientifica' if x in ['CAM2P.3','CAM2P.4','CAM2P.5'] else 'mesoscope' for x in df['equipment_name']]
    df['layer'] = ['deep' if x > 250 else 'shallow' for x in df['imaging_depth']] # NEED TO UPDATE

def make_baseline_figures(VERSION=None,run_params=None, results=None, results_pivoted=None, full_results=None, weights_df = None):
    
    # Analysis Dataframes 
    #####################
    if run_params is None:
        print('loading data')
        run_params, results, results_pivoted, weights_df, full_results = get_analysis_dfs(VERSION)
        print('making figues')

    # Analysis Figures
    #####################
    # Make Nested Model plot (rainbow plot)
    gvt.plot_dropouts(run_params)

    # Make over-fitting figures
    gat.compute_over_fitting_proportion(full_results, run_params) 
    gvt.plot_over_fitting_summary(full_results, run_params)
    gvt.plot_all_over_fitting(full_results, run_params)

    # Make Coding Fraction plots
    gvt.plot_all_coding_fraction(results_pivoted, run_params, metric='fraction')
    gvt.plot_all_coding_fraction(results_pivoted, run_params, metric='magnitude')
    
    # Make Kernel figures
    gvt.all_kernels_evaluation(weights_df,run_params)
    gvt.all_kernels_evaluation(weights_df,run_params,equipment_filter="mesoscope")
    gvt.all_kernels_evaluation(weights_df,run_params,equipment_filter="scientifica")
    gvt.all_kernels_evaluation(weights_df,run_params,session_filter=[1])
    gvt.all_kernels_evaluation(weights_df,run_params,session_filter=[2])
    gvt.all_kernels_evaluation(weights_df,run_params,session_filter=[3])
    gvt.all_kernels_evaluation(weights_df,run_params,session_filter=[4])
    gvt.all_kernels_evaluation(weights_df,run_params,session_filter=[5])
    gvt.all_kernels_evaluation(weights_df,run_params,session_filter=[6])

    # Make Kernel Comparison Figures across sessions
    gvt.plot_all_kernel_comparison(weights_df, run_params, cell_filter='vip',compare=['session'],plot_errors=False)
    gvt.plot_all_kernel_comparison(weights_df, run_params, cell_filter='sst',compare=['session'],plot_errors=False)
    gvt.plot_all_kernel_comparison(weights_df, run_params, cell_filter='slc',compare=['session'],plot_errors=False)
    gvt.plot_all_kernel_comparison(weights_df, run_params, compare=['cre_line'],plot_errors=False)
    gvt.plot_all_kernel_comparison(weights_df, run_params, compare=['cre_line','layer'],plot_errors=False)

