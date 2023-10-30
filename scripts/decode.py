import visual_behavior_glm.decoding as d

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

    d.decode_experiment(args.ophys_experiment_id)
    print('grand finished')
