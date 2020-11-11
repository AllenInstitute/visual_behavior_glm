import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import visual_behavior_glm.GLM_params as glm_params
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



