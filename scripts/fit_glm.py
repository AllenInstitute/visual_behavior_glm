from visual_behavior_glm.src.glm import GLM
import visual_behavior_glm.src.GLM_analysis_tools as gat

import argparse

parser = argparse.ArgumentParser(description='fit glm for experiment')
parser.add_argument(
    '--oeid', 
    type=int, 
    default=0,
    metavar='oeid',
    help='ophys experiment ID'
)

def fit_experiment(oeid):
    glm = GLM(oeid)
    gat.log_results_to_mongo(glm)

if __name__ == '__main__':
    args = parser.parse_args()
    fit_experiment(args.oeid)