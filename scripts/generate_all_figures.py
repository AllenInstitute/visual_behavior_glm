import argparse
import visual_behavior_glm.GLM_params as glm_params
import visual_behavior_glm.GLM_visualization_tools as gvt
import visual_behavior_glm.GLM_analysis_tools as gat
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def generate_all_figures(glm_version):
    '''
    this is the main routine to save figures
    '''
    savepath = get_savepath(glm_version)

    full_results = gat.retrieve_results(results_type='full')
    full_results = unify_underscore_mismatch(full_results)

    full_results = gat.retrieve_results(
        search_dict = {'glm_version': glm_version}, 
        results_type='summary'
    )

    dropout_scores = gat.retrieve_results(search_dict = {'glm_version': glm_version}, results_type='summary')

    # generate a figure that compares variance explained across all sessions
    results_comparison_fig = generate_results_comparison(full_results, savepath=savepath)

    # make a boxplot that compares all dropout scores
    make_and_save_dropout_comparison(dropout_scores=dropout_scores, savepath=savepath)
    

def make_and_save_dropout_comparison(dropout_scores, savepath, figname='dropout_comparison_boxplot'):
    fig, ax = gvt.plot_all_dropout_scores(dropout_scores=dropout_scores)
    if savepath:
        gvt.save_figure(fig, os.path.join(savepath, figname))


def get_savepath(glm_version):
    '''get savepath from run_params'''
    params = glm_params.load_run_json(glm_version)
    return os.path.join(params['output_dir'], 'figures')


def unify_underscore_mismatch(results_in):
    '''
    the convention for the `Full_avg_cv_var_{}` column name changed in V6. 
    This will ensure that the full results dataframe uses the new convention
    '''
    results = results_in.copy()
    for idx, row in results.iterrows():
        for tt in ['train', 'test']:
            if pd.isnull(row['Full__avg_cv_var_{}'.format(tt)]) and pd.notnull(row['Full_avg_cv_var_{}'.format(tt)]):
                results.at[idx, 'Full__avg_cv_var_{}'.format(
                    tt)] = row['Full_avg_cv_var_{}'.format(tt)]
    return results


def generate_results_comparison(results_full, savepath=None, figname='version_comparison'):
    '''
    generates a figure that compares variance explained across all versions
    '''
    fig, ax = gvt.compare_var_explained(
        results_full[results_full['glm_version'] != 'test_fixed_lambda=50'], 
        figsize=(20, 12)
    )
    if savepath:
        gvt.save_figure(figure, os.path.join(savepath, figname))
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='generate all summary figures for a given GLM version'
    )
    parser.add_argument(
        '--glm-version',
        type=str,
        default='',
        metavar='glm_version',
        help='version of GLM to use'
    )
    args = parser.parse_args()

    generate_all_figures(args.glm_version)
