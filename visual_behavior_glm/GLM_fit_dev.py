import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import visual_behavior_glm.GLM_params as glm_params
import visual_behavior_glm.GLM_visualization_tools as gvt
import visual_behavior_glm.GLM_analysis_tools as gat
import visual_behavior_glm.GLM_schematic_plots as gsm
from visual_behavior_glm.glm import GLM
import visual_behavior_glm.GLM_fit_tools as gft


# These functions are useful for quickly wrapping the underlying data streams
# into the glm object, because some analysis functions want the glm object
class dummy_glm:
    a = 'dummy glm'

def make_dummy_glm(fit, run_params, design, session):
    g = type('dummy_glm',(object,),dict(a='dummy glm'))
    g.fit=fit
    g.run_params=run_params
    g.design=design
    g.session=session
    return g

def make_glm(fit, run_params, design, session):
    g = GLM(session.metadata['ophys_experiment_id'],run_params['version'], 
        log_results=False, log_weights=False, recompute=False, 
        use_inputs=True, inputs=[session, fit, design]
        )
    g.run_params = run_params
    return g


if False: # Code snippets for doing analyses. 
    # Experiments for debugging consistency
    #####################
    experiment_table = glm_params.get_experiment_table()
    oeid  = experiment_table.index.values[754]
    oeid1 = experiment_table.index.values[0]
    oeid2 = experiment_table.index.values[154]
    oeid3 = experiment_table.index.values[-1]


    # Make run JSON
    #####################
    VERSION = 1
    src_path = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/GLM/visual_behavior_glm/' 
    glm_params.make_run_json(
        VERSION,
        label='testing',
        username='alex', 
        src_path = src_path, 
        TESTING=True
        )
   


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

    # Deploy all fits on the cluster
    # conda activate visbeh
    # cd ../scripts/
    # vim deploy_glm_fits.sh     # edit to use the proper version
    # ./deploy_glm_fits.sh



    # Load basic results
    #####################

    # Load existing results
    session, fit, design = gft.load_fit_experiment(oeid, run_params)

    # Analyze drop results
    drop_results = gft.build_dataframe_from_dropouts(fit)
    L2_results = gft.L2_report(fit)
    
    # Make GLM object
    g = glm.GLM(
        oeid, 
        VERSION, 
        use_previous_fit=True, 
        log_results=False, 
        log_weights=False
        )
    
    

    # Tools for evaluating model versions
    #####################
    # Get a list of the missing experiments/rois for a specific version
    inventory17 = gat.inventory_glm_version('17_dff_all_L2_optimize_by_session')
    
    # Get a table for GLM versions in the range "vrange"
    inventory_table = gat.build_inventory_table(vrange=[15,20])

    # Compare two model versions
    versions = [x[2:] for x in inventory_table.index.values[-2:]]
    comparison_table,results_combined = gat.get_glm_version_comparison_table(versions)
    gvt.plot_glm_version_comparison(
        comparison_table=comparison_table, 
        versions_to_compare=versions
        )
    gvt.plot_glm_version_comparison_histogram(
        comparison_table=comparison_table, 
        versions_to_compare=versions
        )
    
    # Faster if loading many versions
    results_combined = gat.get_glm_version_summary(versions)
    results_combined = gat.get_glm_version_summary(vrange=[15,20])
    
    # Compare multiple versions
    gvt.compare_var_explained_by_version(results_combined,test_data=True)
    gvt.compare_var_explained_by_version(results_combined,test_data=False)
    gvt.compare_var_explained_by_version(results_combined,cre='Sst-IRES-Cre')
    gvt.compare_var_explained_by_version(
        results_combined,
        cre='Sst-IRES-Cre',
        show_equipment=False
        )   
 
    # Just look at invalid rois
    results_combined = gat.get_glm_version_summary(
        vrange=[15,20], 
        remove_invalid_rois=False, 
        invalid_only=True
        )
    gvt.compare_var_explained_by_version(results_combined)

    # look at overfitting by version
    rc = results_combined.query('Full__avg_cv_var_train > 0.005')
    rc['over_fit'] = rc.apply(lambda x: (x['Full__avg_cv_var_train'] - x['Full__avg_cv_var_test'])/(x['Full__avg_cv_var_train']),axis=1)
    rc.groupby(['cre_line','glm_version'])['over_fit'].mean()

    # Analysis Dataframes 
    #####################

    # results summary # results are experiment/cell/dropout
    results = gat.retrieve_results(
        search_dict={'glm_version':version}, 
        results_type='summary'
        )
 
    # results_pivoted # rows are experiment/cell
    results_pivoted = gat.build_pivoted_results_summary(
        'adj_fraction_change_from_full',
        results_summary=results
        )

    # Full Results
    full_results = gat.retrieve_results(
        search_dict={'glm_version':version}, 
        results_type='full'
        )
 
    # weights_df
    weights_df = gat.build_weights_df(run_params, results_pivoted)



    # Analysis Figures
    #####################

    # Make Nested Model plot (rainbow plot)
    # A couple versions with more or less detail
    schematic_df = gsm.plot_all_dropouts(run_params['version'])
    schematic_df = gsm.plot_high_level_dropouts(run_params['version'])
    schematic_df = gsm.plot_nice_dropouts(run_params['version'])

    # Make the platform paper schematic examples
    oeid = 967008471
    cell_specimen_id = 1086492467
    g=glm.GLM(oeid, version, use_previous_fit=True, log_results=False, log_weights=False)
    gsm.plot_glm_example(g,cell_specimen_id, run_params)

    # Make plot of kernel support
    gvt.plot_kernel_support(g)

    # Make dropout summary figures
    gvt.plot_dropout_summary_population(results,run_params)
    gvt.plot_dropout_individual_population(results,run_params)
    gvt.plot_population_averages(results_pivoted, run_params,add_stats=True)
    gvt.plot_population_averages(results_pivoted, run_params,sharey=False)
    gvt.plot_population_averages(results_pivoted, run_params,
        dropouts_to_show=['behavioral','licks','running','pupil'],extra='_behavioral')
    gvt.plot_population_averages(results_pivoted, run_params,
        dropouts_to_show=['task','hits','misses','all-images'],extra='_task')
    gvt.plot_fraction_summary_population(results_pivoted, run_params)

    # Make over-fitting figures
    # You may need to `mkdir over_fitting_figures` 
    gat.compute_over_fitting_proportion(full_results, run_params) 
    gvt.plot_over_fitting_summary(full_results, run_params)
    gvt.plot_all_over_fitting(full_results, run_params)

    # Make distribution of shuffle results
    gvt.shuffle_analysis(results, run_params)

    # Check impact of different dropout thresholds
    gvt.compare_dropout_thresholds(results)

    # Make Coding Fraction plots
    # You may need to `mkdir coding` 
    gvt.plot_coding_fraction(results_pivoted, run_params, 'omissions') # Example
    gvt.plot_all_coding_fraction(results_pivoted, run_params, metric='fraction') 

    # Make Kernel figures
    # You may need to `mkdir kernels` 
    gvt.kernel_evaluation(weights_df, run_params, 'omissions') # Example
    gvt.all_kernels_evaluation(weights_df,run_params) 
    
    # Make Kernel Comparison Figures
    gvt.plot_kernel_comparison(
        weights_df, 
        run_params, 
        'omissions',
        cell_filter='vip',
        compare=['experience_level'],
        plot_errors=False
        ) 
    gvt.plot_all_kernel_comparison(
        weights_df, 
        run_params,
        cell_filter='vip',
        compare=['session'],
        plot_errors=False
        )
    gvt.plot_kernel_comparison_by_experience(
        weights_df, 
        run_params, 
        'omissions'
        )
    
    # Might need to update
    gvt.plot_perturbation(weights_df, run_params, 'omissions')
    gvt.plot_compare_across_kernels(weights_df, run_params, ['hits','misses'])

def get_analysis_dfs(VERSION):
    run_params = glm_params.load_run_json(VERSION)
    results = gat.retrieve_results(
        search_dict={'glm_version':VERSION},
        results_type='summary'
        )
    results_pivoted = gat.build_pivoted_results_summary(
        'adj_fraction_change_from_full',
        results_summary=results
        )
    weights_df = gat.build_weights_df(run_params, results_pivoted)  
    return run_params, results, results_pivoted, weights_df

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
    print('over fitting figures')
    #full_results = gat.retrieve_results(
    #    search_dict={'glm_version':VERSION},
    #    results_type='full'
    #    )
    gat.compute_over_fitting_proportion(full_results, run_params) 
    gvt.plot_over_fitting_summary(full_results, run_params)
    gvt.plot_all_over_fitting(full_results, run_params)

    # Make Coding Fraction plots
    print('coding fraction figures')
    gvt.plot_all_coding_fraction(results_pivoted, run_params, metric='fraction')
    gvt.plot_all_coding_fraction(results_pivoted, run_params, metric='magnitude')
    
    # Make Kernel figures
    print('kernel evaluation figures')
    gvt.all_kernels_evaluation(weights_df,run_params)
    gvt.all_kernels_evaluation(weights_df,run_params,session_filter=['Familiar'])
    gvt.all_kernels_evaluation(weights_df,run_params,session_filter=['Novel 1'])
    gvt.all_kernels_evaluation(weights_df,run_params,session_filter=['Novel >1'])

    # Make Kernel Comparison Figures across sessions
    print('kernel comparison figures')
    gvt.plot_all_kernel_comparison(weights_df, run_params, cell_filter='vip',compare=['session'],plot_errors=False)
    gvt.plot_all_kernel_comparison(weights_df, run_params, cell_filter='sst',compare=['session'],plot_errors=False)
    gvt.plot_all_kernel_comparison(weights_df, run_params, cell_filter='slc',compare=['session'],plot_errors=False)
    gvt.plot_all_kernel_comparison(weights_df, run_params, compare=['cre_line'],plot_errors=False)
    gvt.plot_all_kernel_comparison(weights_df, run_params, compare=['cre_line','layer'],plot_errors=False)

def dev_ignore():
    gvt.plot_all_kernel_comparison(weights_beh, run_params, compare=['cre_line','strategy'], plot_errors=False) 
    gvt.plot_all_kernel_comparison(weights_beh, run_params, cell_filter='vip', compare=['strategy'], plot_errors=False)
    gvt.plot_all_kernel_comparison(weights_beh, run_params, cell_filter='sst', compare=['strategy'], plot_errors=False)
    gvt.plot_all_kernel_comparison(weights_beh, run_params, cell_filter='slc', compare=['strategy'], plot_errors=False)
    gvt.plot_all_kernel_comparison(weights_beh, run_params, cell_filter='vip', compare=['strategy','layer'], plot_errors=False)
    gvt.plot_all_kernel_comparison(weights_beh, run_params, cell_filter='sst', compare=['strategy','layer'], plot_errors=False)
    gvt.plot_all_kernel_comparison(weights_beh, run_params, cell_filter='slc', compare=['strategy','layer'], plot_errors=False)   

    scatter_by_cell(results_beh, cre_line ='Vip-IRES-Cre',sessions=[1])
    scatter_by_cell(results_beh, cre_line ='Vip-IRES-Cre',sessions=[3])
    scatter_by_cell(results_beh, cre_line ='Vip-IRES-Cre',sessions=[4])
    scatter_by_cell(results_beh, cre_line ='Vip-IRES-Cre',sessions=[6])



