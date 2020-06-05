import visual_behavior_glm.src.GLM_fit_tools as gft
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

if False:
    # Make run JSON
    src_path = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/GLM/visual_behavior_glm/' 
    gft.make_run_json(VERSION, label='Demonstration of make run json',src_path=src_path)

    # Load existing parameters
    run_params = gft.load_run_json(VERSION)

    # To start all experiments on hpc:
    # cd visual_behavior_glm/scripts/
    # python start_glm.py VERSION

    # To run just one session:
    oeid = run_params['ophys_experiment_ids'][-1]
    session, fit, design = gft.fit_experiment(oeid, run_params)

def demonstration():
    # Make demonstration of design kernel, and model structure
    design = gft.DesignMatrix(range(0,20))
    time = np.array(range(1,21))
    time = time/20
    time = time**2
    design.add_kernel(time,2,'time',offset=0)
    events_vec = np.zeros(np.shape(time))
    events_vec[0] = 1
    events_vec[10] = 1
    design.add_kernel(events_vec,3,'discrete',offset=0)
    plt.figure()
    plt.imshow(design.get_X().T)
    plt.xlabel('Weights')
    plt.ylabel('Time')

    W = [1,2,1,.1,.2,.3]
    Y = design.get_X().T @ W
    plt.figure()
    plt.plot(Y, 'k',label='full')
    Y_noIntercept = design.get_X(kernels=['time','discrete']).T@ np.array(W)[1:]
    plt.plot(Y_noIntercept,'r',label='No Intercept')
    Y_noTime = design.get_X(kernels=['intercept','discrete']).T@ np.array(W)[[0,3,4,5]]
    plt.plot(Y_noTime,'b',label='No Time')
    Y_noDiscrete = design.get_X(kernels=['intercept','time']).T@ np.array(W)[0:3]
    plt.plot(Y_noDiscrete,'m',label='No discrete')
    plt.ylabel('dff')
    plt.xlabel('time')
    plt.legend()
    return design.get_X()




