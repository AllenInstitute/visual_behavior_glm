#!/usr/bin/env python

import sys
import json

if __name__ == '__main__':
    name_of_this_file   = sys.argv[0]
    ophys_experiment_id = sys.argv[1]
    run_json            = sys.argv[2]

    # Load JSON for this model version
    with open(run_json,'r') as json_file: 
        run_params = json.load(json_file)
    
    # Import this model version's code
    sys.path.append(run_params['model_freeze_dir'])
    import GLM_fit_tools as gft
    
    # Fit this experiment with given parameters
    gft.fit_experiment(ophys_experiment_id, run_params)


