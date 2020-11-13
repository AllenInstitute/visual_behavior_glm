from visual_behavior_glm.glm import GLM
import visual_behavior_glm.GLM_analysis_tools as gat
import visual_behavior_glm.GLM_visualization_tools as gvt

import visual_behavior.utilities as vbu
import visual_behavior.plotting as vbp
import visual_behavior.data_access.loading as loading
import visual_behavior.visualization.utils as vis_utils

import seaborn as sns
import matplotlib.pyplot as plt

from multiprocessing import Pool

def make_movie_of_top_cells(oeid):
    print('making movies for oeid {}'.format(oeid))
    version = '7_L2_optimize_by_session'
    glm = GLM(int(oeid), version, use_previous_fit=True, log_results=False, log_weights=False)
    glm_results = glm.results.sort_values(by=['Full__avg_cv_var_test'], ascending=False)
    for cell_specimen_id in glm_results.iloc[:2].index.values:
        movie = gvt.GLM_Movie(
            glm,
            cell_specimen_id = cell_specimen_id, 
            start_frame = 30000,
            end_frame = 35000,
            frame_interval = 1,
            destination_folder='/home/dougo/OneDrive/glm_movies',
            fps = 15
        )
        movie.make_movie()

experiment_table = loading.get_filtered_ophys_experiment_table()
table_sample = experiment_table.sample(500, random_state = 1)

for ophys_experiment_id in table_sample.index.values:
    try:
        make_movie_of_top_cells(ophys_experiment_id)
    except Exception as e:
        print('failed on {}'.format(ophys_experiment_id))
        print(e)
        print(' ')