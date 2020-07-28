from visual_behavior_glm.src.glm import GLM
import visual_behavior_glm.src.GLM_analysis_tools as gat
import visual_behavior_glm.src.GLM_visualization_tools as gvt

import visual_behavior.utilities as vbu
import visual_behavior.plotting as vbp
import visual_behavior.data_access.loading as loading
import visual_behavior.visualization.utils as vis_utils

import seaborn as sns
import matplotlib.pyplot as plt

from multiprocessing import Pool

def make_movie_of_top_five_cells(oeid):
    print('making movies for oeid {}'.format(oeid))
    glm = GLM(int(oeid))
    glm_results = glm.results.sort_values(by=['Full_avg_cv_var_test'], ascending=False)
    for cell_specimen_id in glm_results.iloc[:5].index.values:
        movie = gvt.GLM_Movie(
            glm,
            cell_specimen_id = cell_specimen_id, 
            start_frame = 20000,
            end_frame = 40000,
            frame_interval = 15,
            fps = 5
        )
        movie.make_movie()

experiment_table = loading.get_filtered_ophys_experiment_table()
table_sample = experiment_table.sample(500, random_state = 0)

with Pool(10) as pool:
    pool.map(make_movie_of_top_five_cells, table_sample.index.values)