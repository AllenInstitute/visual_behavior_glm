# visual_behavior_glm
Fits a kernel regression model to df/f traces during visual behavior. 

## Fit the model
- Make the run json using `scripts/make_run_json.py`
- Start the run using `scripts/start_glm.py`
- Collect the results across sessions using `scripts/collect_glm.py`

## Model Iteration System
- `scripts/make_run_json.py` saves a copy of current files, as well as a JSON file with run parameters to `../nc-ophys/visual_behavior/ophys_glm/v<iteration-number>/`
