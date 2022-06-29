import visual_behavior_glm.GLM_fit_dev as gfd
import visual_behavior_glm.GLM_visualization_tools as gvt

VERSION = '50_medepalli_test3'
run_params, results, results_pivoted, weights_df = gfd.get_analysis_dfs(VERSION)
gvt.kernel_evaluation(weights_df, run_params, 'omissions', save_results='True', session_filter=['Familiar'])

