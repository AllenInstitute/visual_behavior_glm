# GLM Version descriptions

## regularization schemes
The GLM has been run with three different regularization regularization methods. The version name suffix indicates which method was used. For versions without a suffix, we can assume that the first (fixed lambda) version was used.

* _L2_fixed_lambda=N : a fixed lambda value of N was applied to all cells
* _L2_optimize_by_session : the optimal lambda value was found for each cell by searching on a grid. The best average lambda was calculated across all cells, then applied to all cells (same lambda for all cells)
* _L2_optimize_by_cell : the optimal lambda value was found for each cell by searching on a grid. The best average lambda was calculated for each cell, then applied individually for every cell (potential for different lambda for every cell)

## A number of GLM versions have been run.
Every time a regressor is added, or parameters of a regressor changes, we have incremented the version number. The full set of regressors and associated parameters can be found for a given version by consulting the `run_params` dictionary for that version. The following rougly describes the versions:

* V1 : prototype version. Very limited number of cells run
* V2 : another prototype version. Also limited cells run.
* V3 : First version with many regressors in place. First to be run systematically across all sessions.
* V4 : Similiar parameter set as V3, with decreased support for visual and omission regressors to limit interactions. First version with multiple regularization schemes applied.
* V5 : Included first 50 PCs of face motion energy
* V6 : Modified some regressors slightly. Did some troubleshooting on regularization calculations. Added adjusted dropout scores to account for different levels of support across regressors (i.e, control for fact that some events are rare, some frequent). Added single model dropout scores.