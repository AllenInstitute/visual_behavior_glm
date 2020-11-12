from visual_behavior_glm.GLM_params import make_run_json
import shutil
import os
import argparse

def delete_and_rebuild_run_json(version, label, src_path):
    version_path = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/v_{}/'.format(version)
    if os.path.exists(version_path):
        shutil.rmtree(version_path)
    make_run_json(version,label=label,src_path=src_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rebuild Run json script')
    parser.add_argument(
        '--version', 
        type=str, 
        default='9a_L2_optimize_by_session',
        metavar='model_version',
        help='model version'
    )
    parser.add_argument(
        '--label', 
        type=str, 
        default='model as of 10/22/2020, include model predicted licks and grooming',
        metavar='model_label',
        help='model label'
    )
    parser.add_argument(
        '--src-path', 
        type=str, 
        default='/home/dougo/Code/visual_behavior_glm/',
        metavar='src_path',
        help='folder where code lives'
    )
    
    args = parser.parse_args()

    delete_and_rebuild_run_json(args.version, args.label, args.src_path)
