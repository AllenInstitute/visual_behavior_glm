from visual_behavior_glm.glm import GLM
import visual_behavior_glm.GLM_analysis_tools as gat

versions_to_compare = [
    '10a_L2_optimize_by_session',
    '11a_L2_optimize_by_session'
]

comparison_table = gat.get_glm_version_comparison_table(versions_to_compare)

def version_diff(row):
    try:
        return (row['10a_L2_optimize_by_session'] - row['11a_L2_optimize_by_session'])/row['10a_L2_optimize_by_session']
    except ZeroDivisionError:
        return None
comparison_table['version_diff'] = comparison_table.apply(version_diff, axis=1)

big_droppers = (
    comparison_table[
        (comparison_table[versions_to_compare[0]] > 0.1)
        &(comparison_table[versions_to_compare[1]] > 0)
    ]
    .dropna(subset=versions_to_compare)[versions_to_compare + ['version_diff','identifier','cre_line']]
    .sort_values(by='version_diff', ascending=False)
    .iloc[:20]
)

for idx, row in big_droppers.iterrows():
    oeid, csid = row['identifier'].split('_')
    for version in versions_to_compare:
        try:
            glm = GLM(
                oeid, 
                version, 
                use_previous_fit=True,  # if True, uses cached fit if available (False by default)
                log_results=False, # if True, logs fit results to mongo database (True by default)
                log_weights=False, # if True, logs weights to mongo database (True by default)
                NO_DROPOUTS=False, # if True, does not perform dropout analysis (False by default)
                TESTING=False, # if True, fits only the first 6 cells in the experiment (False by default)
            )

            glm.movie(
                cell_specimen_id=csid,
                action='make_movie',
                frame_interval=50,
                fps=5,
                destination_folder = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/events_dff_comparison_videos'
            )
        except:
            pass
