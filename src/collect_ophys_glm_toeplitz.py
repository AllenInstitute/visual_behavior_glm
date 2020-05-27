import json
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import sys
from json import JSONDecodeError
from allensdk.brain_observatory.behavior.behavior_project_cache import BehaviorProjectCache
from matplotlib import pyplot as plt
import argparse
import mlflow


parser = argparse.ArgumentParser(description='GLM Fitter')
parser.add_argument('--run-json', type=str, default='',
                    metavar='/path/to/run/json',
                    help='output data path')
args = parser.parse_args()

with open(args.run_json, 'r') as json_file:
    run_json = json.load(json_file)

cache = BehaviorProjectCache.from_lims(manifest=run_json['manifest'])
ophys_sessions = run_json['ophys_sessions']
output_dir = run_json['output_dir']

full_cv_var_key = 'full_test'
all_dfs = []
for osid in tqdm(ophys_sessions):
    fn = 'osid_{}.json'.format(osid)
    output_full_path = os.path.join(output_dir, fn)
    try:
        with open(output_full_path, 'r') as json_file:
            this_dict = json.load(json_file)

            ophys_session_id = this_dict.pop('ophys_session_id')
            this_df = pd.DataFrame(this_dict)
            this_df['ophys_session_id']=ophys_session_id

            #  if np.any(pd.isnull(this_df[full_cv_var_key])):
            #      nan_frac = np.mean(pd.isnull(this_df[full_cv_var_key]))
            #      print('{} NAN values for var explained in file: {}'.format(nan_frac, output_full_path))
            #  else:
            all_dfs.append(this_df)
    except (FileNotFoundError, JSONDecodeError) as e:
        #  print('problem for session {}, {}'.format(osid, e))
        pass

dfs_with_csid = [df.reset_index().rename(columns={'index':'cell_specimen_id'}) for df in all_dfs]
full_df = pd.concat(dfs_with_csid, ignore_index=True)
df_save_full_path = os.path.join(output_dir, 'full_df.hdf')
manifest_sessions = cache.get_session_table()

## Join on session metadata
full_df = full_df.merge(manifest_sessions, on='ophys_session_id', how='inner')
full_df.to_hdf(df_save_full_path, key='df')
print("saved: {}".format(df_save_full_path))

#  for key, val in run_params: 
#      mlflow.log_param(key, val)

# Make some figures and log them as artifacts
## Histogram of var explained
#  plt.clf()
#  bin_edges = np.linspace(-0.1, 0.8, 200)
#  plt.hist(full_df.query('responsive==1.0')[full_cv_var_key],
#           bins=bin_edges, histtype='step', label='responsive')
#  plt.hist(full_df.query('responsive==0.0')[full_cv_var_key],
#           bins=bin_edges, histtype='step', label='non-responsive')
#  plt.yscale('log') 
#  plt.ylabel('# cells')
#  plt.xlabel('cross-validated variance explained')
#  plt.legend()
#  fig_fn = os.path.join(output_dir, 'hist_var_explained.png')
#  plt.savefig(fig_fn)
#  print("saved: {}".format(fig_fn))
#  #  mlflow.log_artifact(fig_fn)
#  
#  ## Mean for each group
#  mean_responsive = full_df.query('responsive==1.0')[full_cv_var_key].mean()
#  print("Mean var explained, responsive cells: {}".format(mean_responsive))
#  #  mlflow.log_metric('mean_responsive_cv_var_explained', mean_responsive)
#  mean_non_responsive = full_df.query('responsive==0.0')[full_cv_var_key].mean()
#  print("Mean var explained, non-responsive cells: {}".format(mean_non_responsive))
#  #  mlflow.log_metric('mean_non_responsive_cv_var_explained', mean_non_responsive)

## Boxplot for all, reliable, unreliable
#TODO: This is pretty much all broken because the new manifest doesn't have the data we need

'''
plt.clf()
y_allcells = full_df['cv_var_explained'].values
y_responsive = full_df.query('responsive==1.0')['cv_var_explained'].values
y_nonresponsive = full_df.query('responsive==0.0')['cv_var_explained'].values
y_above2 = full_df.query('cv_var_explained>0.02')['cv_var_explained'].values
y_slc = full_df.query('full_genotype=="Slc17a7-IRES2-Cre/wt;Camk2a-tTA/wt;Ai93(TITL-GCaMP6f)/wt"')['cv_var_explained'].values
y_vip = full_df.query('full_genotype=="Vip-IRES-Cre/wt;Ai148(TIT2L-GC6f-ICL-tTA2)/wt"')['cv_var_explained'].values
#  y_175 =full_df.query('imaging_depth==175')['cv_var_explained'].values
#  y_375=full_df.query('imaging_depth==375')['cv_var_explained'].values
y_a1=full_df.query('stage_name=="OPHYS_1_images_A"')['cv_var_explained'].values
y_a2=full_df.query('stage_name=="OPHYS_2_images_A_passive"')['cv_var_explained'].values
y_a3=full_df.query('stage_name=="OPHYS_3_images_A"')['cv_var_explained'].values
y_b1=full_df.query('stage_name=="OPHYS_4_images_B"')['cv_var_explained'].values
y_b2=full_df.query('stage_name=="OPHYS_5_images_B_passive"')['cv_var_explained'].values
y_b3=full_df.query('stage_name=="OPHYS_6_images_B"')['cv_var_explained'].values
datalist = [y_allcells, y_responsive, y_nonresponsive,
            y_above2, y_slc, y_vip,
            #  y_175,y_375,
            y_a1, y_a2, y_a3, y_b1, y_b2, y_b3]
datalist = [x[~np.isnan(x)] for x in datalist]
xlabels = ['all', 'responsive', 'nonresponsive', 'above 2%', 'Slc17a', 
           'Vip',
           #  '175','375',
           'A1', 'A2', 'A3', 'B1', 'B2', 'B3']
plt.boxplot(datalist, showfliers=False)
plt.gca().set_xticks(1+np.arange(len(xlabels)))
plt.gca().set_xticklabels(xlabels, rotation=60)
plt.ylabel('cross validated variance explained')
plt.tight_layout()
fig_fn = os.path.join(output_dir, 'var_explained_per_condition.png')
plt.savefig(fig_fn)
print("saved: {}".format(fig_fn))
'''

#  mlflow.log_artifact(fig_fn)

#  elif case==1:
#      output_dir = '/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/20191223_cv_reg_param_search'
#      all_lambdas = []
#      for osid in tqdm(ophys_sessions):
#          fn = 'osid_{}.json'.format(osid)
#          output_full_path = os.path.join(output_dir, fn)
#          try:
#              with open(output_full_path, 'r') as json_file:
#                  this_dict = json.load(json_file)
#                  all_lambdas.append(this_dict['best_lambda'])
#          except (FileNotFoundError, JSONDecodeError):
#              print('problem for session {}'.format(osid))
