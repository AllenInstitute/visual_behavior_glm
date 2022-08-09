import visual_behavior_glm.GLM_params as glm_params
import GLM_analysis_tools as gat
import pandas as pd
import os
import time
import matplotlib.pyplot as plt

version = '57_medepalli_omission_specific_analysis_pre_post'
run_params = glm_params.load_run_json(version)
oeid = 967008471

start = time.time()
fit = gat.load_fit_pkl(run_params, oeid)
end = time.time()

t = fit['fit_trace_timestamps']
y = fit['fit_trace_arr'].values

fig, ax = plt.plot(t, y)
ax.set_xlim(1790, 1798)
ax.set_ylabel('df/f response (events)')
ax.set_xlabel('time (s)')
plt.savefig(run_params['figure_dir'] + '/vis_gt.svg')
