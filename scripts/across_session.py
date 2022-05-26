import visual_behavior_glm.GLM_across_session as gas

import argparse

parser = argparse.ArgumentParser(description='compute across session dropout')
parser.add_argument(
    '--cell', 
    type=int, 
    default=0,
    metavar='cell',
    help='cell_specimen_id'
)
parser.add_argument(
    '--version', 
    type=str, 
    default='',
    metavar='glm_version',
    help='glm_version'
)

if __name__ == '__main__':
    args = parser.parse_args()
    data, score_df = gas.across_session_normalization(args.cell,args.version)
