# visual_behavior_glm
Fits a kernel regression model to df/f traces during visual behavior. 

## Defining new kernels/regressors
- Adding a new kernel requires changes in two places.
- Add the kernel parameters to `make_run_json()` as a dictionary with keys `length` and `offset`
- Define the event times in `add_kernel_by_label()`

## Fit the model
- Make the run json using `delete_rebuild_run_json.py` in  `scripts` with `python delete_rebuild_run_json.py --version <version> --label <a descriptive label> --src-path <path_to_source_code>`
- Start the run for a single session with `python scripts/fit_glm.py --oied <ophys_experiment_id> --version <version>`
- Start the run for all sessions at the command line on hpc-login with `python scripts/deploy_fits.py --glm-version <version> --env <name_of_conda_environment>`
- Collect the results across sessions using `retrieve_results(glm_version=<version>)` from `src/GLM_analysis_tools.py`

## Model Iteration System
- `delete_rebuild_run_json.py` saves a copy of current files, as well as a JSON file with run parameters to `../nc-ophys/visual_behavior/ophys_glm/v_<version>/`

