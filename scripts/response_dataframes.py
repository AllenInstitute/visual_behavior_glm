import visual_behavior_glm.build_dataframes as bd

import argparse

parser = argparse.ArgumentParser(description='build response dataframe for experiment')
parser.add_argument(
    '--ophys_experiment_id', 
    type=int, 
    default=0,
    metavar='oeid',
    help='ophys_experiment_id'
)

if __name__ == '__main__':
    args = parser.parse_args()
    session = bd.load_data(args.oeid)
    bd.build_response_df_experiment(session)
