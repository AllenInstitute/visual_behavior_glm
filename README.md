# visual_behavior_glm
Fits a kernel regression model to df/f traces during visual behavior. 

## Fit the model
- Make the run json using `make_run_json(<version>)` in  `src/GLM_fit_tools.py`
- Start the run using `scripts/start_glm.py` with `python start_glm.py <version>`
- Collect the results across sessions using `scripts/collect_glm.py`

## Model Iteration System
- `make_run_json(<version>)` saves a copy of current files, as well as a JSON file with run parameters to `../nc-ophys/visual_behavior/ophys_glm/v_<version>/`
- use as: `make_run_json(1, label='Brief description of version #1')`
- load the json with: `run_params = make_run_json(1)`
- check which sessions have fits with: `experiment_table = check_run_fits(1)['GLM_fit']`

## Defining new kernels/regressors
- Adding a new kernel requires changes in two places.
- Add the kernel parameters to `make_run_json()` as a dictionary with keys `length` and `offset`
- Define the event times in `add_kernel_by_label()`
