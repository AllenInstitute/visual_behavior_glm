import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import visual_behavior_glm.src.GLM_params as glm_params
plt.ion()

def make_test_json():
    # Make run JSON
    VERSION = 1
    src_path = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/GLM/visual_behavior_glm/' 
    glm_params.make_run_json(VERSION,label='testing',username='alex', src_path = src_path, TESTING=True)
    
if False:
    # import_dir = self.run_params['model_freeze_dir'].rstrip('/')
    # module_name = 'GLM_fit_tools'
    # file_path = os.path.join(import_dir, module_name+'.py')
    # print('importing {} from {}'.format(module_name, file_path))

    # spec = importlib.util.spec_from_file_location(module_name, file_path)
    # gft = importlib.util.module_from_spec(spec)
    # sys.modules[module_name] = gft
    # spec.loader.exec_module(gft)

    # To run just one session:
    run_params = glm_params.load_run_json('4_L2_optimize_by_cell')
    fit = gat.load_fit_pkl(run_params, oeid)
    oeid = run_params['ophys_experiment_ids'][-1]
    session, fit, design = gft.fit_experiment(oeid, run_params)
    drop_results = gft.build_dataframe_from_dropouts(fit)
    L2_results = gft.L2_report(fit)


def demonstration():
    # Make demonstration of design kernel, and model structure
    fit_demo = {}
    fit_demo['dff_trace_timestamps'] = range(0,20)
    fit_demo['ophys_frame_rate'] = 1
    design = gft.DesignMatrix(fit_demo)
    time = np.array(range(1,21))
    time = time/20
    time = time**2
    design.add_kernel(np.ones(np.shape(time)),1,'intercept',offset=0)
    design.add_kernel(time,2,'time',offset=0)
    events_vec = np.zeros(np.shape(time))
    events_vec[0] = 1
    events_vec[10] = 1
    design.add_kernel(events_vec,3,'discrete-post',offset=0)
    design.add_kernel(events_vec,3,'discrete-pre',offset=-3)
    plt.figure()
    plt.imshow(design.get_X())
    plt.xlabel('Weights')
    plt.ylabel('Time')

    W =[1,2,1,.1,.2,.3,.4,.5,.6]
    Y = design.get_X().values @ W
    plt.figure()
    plt.plot(Y, 'k',label='full')
    Y_noIntercept = design.get_X(kernels=['time','discrete-post','discrete-pre']).values@ np.array(W)[1:]
    plt.plot(Y_noIntercept,'r',label='No Intercept')
    Y_noTime = design.get_X(kernels=['intercept','discrete-post','discrete-pre']).values@ np.array(W)[[0,3,4,5,6,7,8]]
    plt.plot(Y_noTime,'b',label='No Time')
    Y_noDiscrete = design.get_X(kernels=['intercept','time']).values@ np.array(W)[0:3]
    plt.plot(Y_noDiscrete,'m',label='No discrete')
    plt.ylabel('dff')
    plt.xlabel('time')
    plt.legend()
    return design.get_X()




