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
* V7 : Adds a significant number of dropout groups that now form a series of nested models. Slightly modifies the times of some of the kernels. Computes an over-fitting proportion. Adds a threshold that only adds a discrete kernel if there are at least 5 of that event in the entire session. 
* V8a : Explicitly defines hit/miss/false-alarm/correct-reject regressors, but removes reward and change regressors
* V8b : Implements reward and change regressors, removes hit/miss/false-alarm/correct-reject regressors
* V9a : based on V8a, uses a licks kernel
* V9b : based on V8a, uses a lick-bouts kernel (but not individual licks)
* V9c : based on V8a, uses a lick-model kernel
* V9d : based on V8a, uses a lick-model kernel and a groom-model kernel
* V10a : based on V9a, has a kernel for each image with params: length = 0.767, offset = -0.25
* V10b : based on V9a, has a kernel for each image with params: length = 0.25, offset = -0.25, plus image_expectation with params: length = 0.5, offset = 0
* V10c : based on V9a, has a kernel for each image with params: length = 0.767, offset = 0