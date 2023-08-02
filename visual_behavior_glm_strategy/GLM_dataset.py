import os
import scipy
import pickle
import numpy as np
import matplotlib.pyplot as plt
import visual_behavior_glm_strategy.GLM_fit_dev as gfd

# To save kernel dictionaries for a new glm version:
# run_params, results, results_pivoted, weights_df, full_results = utils.load_glm_data(glm_version)
# utils.save_data_summary(run_params, weights_df)

# To generate a dataset:
# dataset = utils.make_dataset(image_set, glm_version,normalize)

# To generate a new composite:
# labels = utils.define_composite(num_images=20, omissions=[5,16], changes=[10])
# dataset['composite'] = utils.make_composite(dataset,labels)

# Data directory
FILEPATH = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/' 
GLM_VERSION = '12_dff_L2_optimize_by_session'

def load_glm_data(glm_version):
    '''
        loads the weight data from the GLM database
        glm_version: a string with the name of the model version to use
    '''
    run_params, results, results_pivoted, weights_df, full_results = gfd.get_analysis_dfs(glm_version)
    return run_params, results, results_pivoted, weights_df, full_results

def save_data_summary(run_params, weights_df,normalize='none',other=None):
    '''
        Saves out familiar and novel kernel dictionaries for the average image and omissions
        run_params: glm parameter dictionary
        weights_df: dataframe of kernel weights
    '''
    _,_,_,_,_,_,_,_,_,_,normalization_amplitude = process_data(run_params, weights_df, 'image',normalize=normalize,other=other)
    _ = process_data(run_params, weights_df, 'omissions',normalize=normalize,normalization_amplitude = normalization_amplitude,other=other)
    _ = process_data(run_params, weights_df, 'hits',normalize=normalize, normalization_amplitude = normalization_amplitude,other=other)

def process_data(run_params, weights_df,kernel,normalize='max_by_cre',normalization_amplitude=None,other=None):
    '''
        Computes the average kernel for each cre line, handles normalization, and
        interpolation onto scientifica timebase. Saves the kernel dictionaries.

        run_params: glm parameter dictionary
        weights_df: dataframe of kernel weights
        kernel: a string name of the kernel to save. use "image" for the average image 
        normalize: a string saying the normalization method to use
            'max_by_cell' normalizes each cell's kernel to have max(abs()) == 1
            'max_by_cre'  normalizes each cell's kernels such that the image kernel has max(abs()) == 1, and other kernels are on the same scale
            'max_by_kernel' normalizes each cre-lines average kernel to have max(abs()) == 1 
            'none' performs no normalization
        
    '''    

    do_passive= (kernel in ['image','omissions']) & (other not in ['visual','timing'])
    # These are mesoscope sessions with the wrong sampling rate
    problem_sessions = [962045676, 1048363441,1050231786,1051107431,1051319542,1052096166,1052512524,1052752249,1049240847,1050929040,1052330675,1050597678,1056238781,1056065360]

    # Filter to area VISp, novel, and familar
    familiar_weights = weights_df.query('(ophys_session_id not in @problem_sessions) & (targeted_structure == "VISp") & (session_number in ["1","3"]) & (layer == "shallow")').copy() # about 10k ROIs
    fpassive_weights = weights_df.query('(ophys_session_id not in @problem_sessions) & (targeted_structure == "VISp") & (session_number in ["2"]) & (layer == "shallow")').copy() # about 10k ROIs
    novel_weights = weights_df.query('(ophys_session_id not in @problem_sessions) & (targeted_structure == "VISp") & (prior_exposures_to_image_set == 0) & (layer == "shallow")').copy() # about 3.7k ROIs
    novelp_weights = weights_df.query('(ophys_session_id not in @problem_sessions) & (targeted_structure == "VISp") & (session_number in ["4","6"]) & (prior_exposures_to_image_set > 0 ) & (layer == "shallow")').copy() # about 5k 
    nppassive_weights = weights_df.query('(ophys_session_id not in @problem_sessions) & (targeted_structure == "VISp") & (session_number in ["5"]) & (prior_exposures_to_image_set > 0 ) & (layer == "shallow")').copy() # about 5k 

    # remove NaNs
    if kernel != 'image':
        familiar_weights = familiar_weights[familiar_weights[kernel+'_weights'].notnull()].copy()
        if do_passive:
            fpassive_weights = fpassive_weights[fpassive_weights[kernel+'_weights'].notnull()].copy()
        novel_weights = novel_weights[novel_weights[kernel+'_weights'].notnull()].copy()
        novelp_weights = novelp_weights[novelp_weights[kernel+'_weights'].notnull()].copy()
        if do_passive:
            nppassive_weights = nppassive_weights[nppassive_weights[kernel+'_weights'].notnull()].copy()

    # Get time vectors
    sci_time, meso_time = get_time_vecs(kernel, run_params)

    # Interpolate onto scientifica time base
    if kernel == "image":
        # Interpolate all 8 images
        for k in range(0,8):           
            familiar_weights[kernel+str(k)+'_weights'] = [
                x if len(x) == len(sci_time) else 
                scipy.interpolate.interp1d(
                    meso_time, x, fill_value = "extrapolate", 
                    bounds_error=False)(sci_time) 
                for x in familiar_weights[kernel+str(k)+'_weights']]
            if do_passive:
                fpassive_weights[kernel+str(k)+'_weights'] = [
                    x if len(x) == len(sci_time) else 
                    scipy.interpolate.interp1d(
                        meso_time, x, fill_value = "extrapolate", 
                        bounds_error=False)(sci_time) 
                    for x in fpassive_weights[kernel+str(k)+'_weights']]
            novel_weights[kernel+str(k)+'_weights'] = [
                x if len(x) == len(sci_time) else 
                scipy.interpolate.interp1d(
                    meso_time, x, fill_value = "extrapolate", 
                    bounds_error=False)(sci_time) 
                for x in novel_weights[kernel+str(k)+'_weights']]
            novelp_weights[kernel+str(k)+'_weights'] = [
                x if len(x) == len(sci_time) else 
                scipy.interpolate.interp1d(
                    meso_time, x, fill_value = "extrapolate", 
                    bounds_error=False)(sci_time) 
                for x in novelp_weights[kernel+str(k)+'_weights']]
            if do_passive:
                nppassive_weights[kernel+str(k)+'_weights'] = [
                    x if len(x) == len(sci_time) else 
                    scipy.interpolate.interp1d(
                        meso_time, x, fill_value = "extrapolate", 
                        bounds_error=False)(sci_time) 
                    for x in nppassive_weights[kernel+str(k)+'_weights']]

        # Average 8 images together
        familiar_weights['image_weights'] = familiar_weights.apply(
            lambda x: np.mean([
                x['image0_weights'],
                x['image1_weights'],
                x['image2_weights'],
                x['image3_weights'],
                x['image4_weights'],
                x['image5_weights'],
                x['image6_weights'],
                x['image7_weights']],
                axis=0),axis=1)
        if do_passive:
            fpassive_weights['image_weights'] = fpassive_weights.apply(
                lambda x: np.mean([
                    x['image0_weights'],
                    x['image1_weights'],
                    x['image2_weights'],
                    x['image3_weights'],
                    x['image4_weights'],
                    x['image5_weights'],
                    x['image6_weights'],
                    x['image7_weights']],
                    axis=0),axis=1)
        novel_weights['image_weights'] = novel_weights.apply(
            lambda x: np.mean([
                x['image0_weights'],
                x['image1_weights'],
                x['image2_weights'],
                x['image3_weights'],
                x['image4_weights'],
                x['image5_weights'],
                x['image6_weights'],
                x['image7_weights']],
                axis=0),axis=1)

        novelp_weights['image_weights'] = novelp_weights.apply(
            lambda x: np.mean([
                x['image0_weights'],
                x['image1_weights'],
                x['image2_weights'],
                x['image3_weights'],
                x['image4_weights'],
                x['image5_weights'],
                x['image6_weights'],
                x['image7_weights']],
                axis=0),axis=1)
        if do_passive:
                nppassive_weights['image_weights'] = nppassive_weights.apply(
                lambda x: np.mean([
                    x['image0_weights'],
                    x['image1_weights'],
                    x['image2_weights'],
                    x['image3_weights'],
                    x['image4_weights'],
                    x['image5_weights'],
                    x['image6_weights'],
                    x['image7_weights']],
                    axis=0),axis=1)
    else:
        # Interpolate this kernel
        familiar_weights[kernel+'_weights'] = [
            x if len(x) == len(sci_time) else 
            scipy.interpolate.interp1d(
                meso_time, x, fill_value = "extrapolate", 
                bounds_error=False)(sci_time) 
            for x in familiar_weights[kernel+'_weights']]
        if do_passive:
            fpassive_weights[kernel+'_weights'] = [
                x if len(x) == len(sci_time) else 
                scipy.interpolate.interp1d(
                    meso_time, x, fill_value = "extrapolate", 
                    bounds_error=False)(sci_time) 
                for x in fpassive_weights[kernel+'_weights']]
        novel_weights[kernel+'_weights'] = [
            x if len(x) == len(sci_time) else 
            scipy.interpolate.interp1d(
                meso_time, x, fill_value = "extrapolate", 
                bounds_error=False)(sci_time) 
            for x in novel_weights[kernel+'_weights']]
        novelp_weights[kernel+'_weights'] = [
            x if len(x) == len(sci_time) else 
            scipy.interpolate.interp1d(
                meso_time, x, fill_value = "extrapolate", 
                bounds_error=False)(sci_time) 
            for x in novelp_weights[kernel+'_weights']]
        if do_passive:
            nppassive_weights[kernel+'_weights'] = [
                x if len(x) == len(sci_time) else 
                scipy.interpolate.interp1d(
                    meso_time, x, fill_value = "extrapolate", 
                    bounds_error=False)(sci_time) 
                for x in nppassive_weights[kernel+'_weights']]
 
    # Make average kernel dictionaries
    f_kernel_dict = {}
    f_kernel_dict['time'] = sci_time
    fp_kernel_dict = {}
    fp_kernel_dict['time'] = sci_time
    n_kernel_dict = {}
    n_kernel_dict['time'] = sci_time
    p_kernel_dict = {}
    p_kernel_dict['time'] = sci_time
    pp_kernel_dict = {}
    pp_kernel_dict['time'] = sci_time
    cres = ['Slc17a7-IRES2-Cre','Sst-IRES-Cre','Vip-IRES-Cre']
    for cre in cres:
        # split by cre line and average
        temp = familiar_weights.query('cre_line == @cre')
        f_kernel_dict[cre] = np.mean(np.vstack(temp[kernel+'_weights']),0)
        f_kernel_dict[cre+'_std'] = np.std(np.vstack(temp[kernel+'_weights']),0)
        f_kernel_dict[cre+'_n'] = len(temp) 
        f_kernel_dict[cre+'_sem'] = f_kernel_dict[cre+'_std']/np.sqrt(f_kernel_dict[cre+'_n'])

        if do_passive:
            temp = fpassive_weights.query('cre_line == @cre')
            fp_kernel_dict[cre] = np.mean(np.vstack(temp[kernel+'_weights']),0)
            fp_kernel_dict[cre+'_std'] = np.std(np.vstack(temp[kernel+'_weights']),0)
            fp_kernel_dict[cre+'_n'] = len(temp) 
            fp_kernel_dict[cre+'_sem'] = fp_kernel_dict[cre+'_std']/np.sqrt(fp_kernel_dict[cre+'_n'])

        temp = novel_weights.query('cre_line == @cre')
        n_kernel_dict[cre] = np.mean(np.vstack(temp[kernel+'_weights']),0)
        n_kernel_dict[cre+'_std'] = np.std(np.vstack(temp[kernel+'_weights']),0)
        n_kernel_dict[cre+'_n'] = len(temp) 
        n_kernel_dict[cre+'_sem'] = n_kernel_dict[cre+'_std']/np.sqrt(n_kernel_dict[cre+'_n'])

        temp = novelp_weights.query('cre_line == @cre')
        p_kernel_dict[cre] = np.mean(np.vstack(temp[kernel+'_weights']),0)
        p_kernel_dict[cre+'_std'] = np.std(np.vstack(temp[kernel+'_weights']),0)
        p_kernel_dict[cre+'_n'] = len(temp) 
        p_kernel_dict[cre+'_sem'] = p_kernel_dict[cre+'_std']/np.sqrt(p_kernel_dict[cre+'_n'])

        if do_passive:
            temp = nppassive_weights.query('cre_line == @cre')
            pp_kernel_dict[cre] = np.mean(np.vstack(temp[kernel+'_weights']),0)
            pp_kernel_dict[cre+'_std'] = np.std(np.vstack(temp[kernel+'_weights']),0)
            pp_kernel_dict[cre+'_n'] = len(temp)
            pp_kernel_dict[cre+'_sem'] = pp_kernel_dict[cre+'_std']/np.sqrt(pp_kernel_dict[cre+'_n'])

    # Make metadata dictionary
    meta= {}
    meta['glm_version'] = run_params['version']
    meta['area'] = 'VISp'
    meta['normalization'] = normalize
    if 'events' in run_params['version']:
        meta['data_type'] = 'events'
    else:
        meta['data_type'] = 'dff'
    meta['kernel'] = kernel
    f_kernel_dict['meta'] = meta
    fp_kernel_dict['meta'] = meta
    n_kernel_dict['meta'] = meta
    p_kernel_dict['meta'] = meta
    pp_kernel_dict['meta'] = meta

    normalization_amplitudes = {}

    # save out dictionaries
    f_filename = get_filepath(kernel, normalize, 'familiar',area='VISp', glm_version = run_params['version'],other=other) 
    fp_filename = get_filepath(kernel, normalize, 'fpassive',area='VISp', glm_version = run_params['version'],other=other) 
    n_filename = get_filepath(kernel, normalize, 'novel',   area='VISp', glm_version = run_params['version'],other=other)
    p_filename = get_filepath(kernel, normalize, 'novelp',   area='VISp', glm_version = run_params['version'],other=other)
    pp_filename = get_filepath(kernel, normalize, 'nppassive',   area='VISp', glm_version = run_params['version'],other=other)
    
    save_kernels(f_kernel_dict, f_filename)   
    if do_passive:
        save_kernels(fp_kernel_dict, fp_filename)    
    save_kernels(n_kernel_dict, n_filename)   
    save_kernels(p_kernel_dict, p_filename)    
    if do_passive:
        save_kernels(pp_kernel_dict, pp_filename)    
    return familiar_weights, fpassive_weights, novel_weights, novelp_weights, nppassive_weights, f_kernel_dict,fp_kernel_dict, n_kernel_dict,p_kernel_dict,pp_kernel_dict, normalization_amplitudes

def get_time_vecs(kernel, run_params):
    '''
        Returns the scientifica and mesoscope timestamps for this kernel
    '''
    if kernel == "image":
        kernel = "image0"
    sci_time = np.arange(run_params['kernels'][kernel]['offset'], run_params['kernels'][kernel]['offset'] + run_params['kernels'][kernel]['length'],1/31)
    sci_time = np.round(sci_time,2)
    meso_time = np.arange(run_params['kernels'][kernel]['offset'], run_params['kernels'][kernel]['offset'] + run_params['kernels'][kernel]['length'],1/10.725)
    if kernel == 'omissions':
        meso_time = np.concatenate([meso_time,[meso_time[-1]+np.diff(meso_time)[0]]])
    if kernel == 'hits':
        meso_time = np.concatenate([meso_time,[meso_time[-1]+np.diff(meso_time)[0]],[meso_time[-1]+2*np.diff(meso_time)[0]]])
    return sci_time, meso_time

def get_filepath(kernel, normalize, image_set, area='VISp', glm_version=None,other=None):
    '''
        Determines the filepath for where to save this kernel dictionary
    '''
    if glm_version is None:
        glm_version = GLM_VERSION

    if not os.path.isdir(FILEPATH + 'v_'+glm_version +'/dataset'):
        os.mkdir(FILEPATH + 'v_'+glm_version +'/dataset') 

    if 'events' in glm_version:
        data_type = 'events'
    else:
        data_type = 'dff'
    if other is None:
        other = ''
    if image_set == 'familiar': 
        return FILEPATH +'v_'+glm_version+'/dataset/'+ kernel+'_normalized_by_'+normalize+'_sessions_1_3_area_'+area+'_GLM_v'+glm_version[0:2] +'_'+data_type+'_'+other+'.pkl'       
    elif image_set == 'fpassive':
        return FILEPATH +'v_'+glm_version+'/dataset/'+ kernel+'_normalized_by_'+normalize+'_sessions_2_area_'+area+'_GLM_v'+glm_version[0:2] +'_'+data_type+'_'+other+'.pkl'       
    elif image_set == 'novel':
        return FILEPATH +'v_'+glm_version+'/dataset/'+ kernel+'_normalized_by_'+normalize+'_sessions_4_area_'+area+'_GLM_v'+glm_version[0:2] +'_'+data_type+'_'+other+'.pkl'       
    elif image_set == 'nppassive':
        return FILEPATH +'v_'+glm_version+'/dataset/'+ kernel+'_normalized_by_'+normalize+'_sessions_5_area_'+area+'_GLM_v'+glm_version[0:2] +'_'+data_type+'_'+other+'.pkl'       
    else:
        return FILEPATH +'v_'+glm_version+'/dataset/'+ kernel+'_normalized_by_'+normalize+'_sessions_6_area_'+area+'_GLM_v'+glm_version[0:2] +'_'+data_type+'_'+other+'.pkl'       
 
def save_kernels(kernel_dict, filename):
    '''
        Pickle wrapper function
    '''
    with open(filename,'wb') as handle:
        pickle.dump(kernel_dict, handle)
    return

def load_kernels(filepath): 
    '''
        Pickle wrapper function
    '''
    filetemp = open(filepath,'rb')
    data    = pickle.load(filetemp)
    filetemp.close()
    return data

def make_dataset(image_set='familiar', glm_version=None, area='VISp',normalize='max_by_cre',other=None):
    '''
        Loads the kernel dictionaries for images, omissions, and a composite
        image_set: a string, should be 'familiar' or 'novel'
        glm_version: a string, name of model version to use
    '''
    do_passive = image_set in ['familiar','novel','novelp']
    if glm_version is None:
        glm_version = GLM_VERSION
    images =    load_kernels(get_filepath('image',normalize, image_set,area, glm_version,other))
    omissions = load_kernels(get_filepath('omissions',normalize, image_set,area, glm_version,other))
    dataset = {
        'images': images,
        'omissions':omissions
        }
    if do_passive:
        changes = load_kernels(get_filepath('hits',normalize, image_set,area, glm_version,other))
        dataset['changes'] = changes
    dataset['composite_images']   = make_composite(dataset, labels=define_composite())
    dataset['composite_omission'] = make_composite(dataset, labels=define_composite(omissions=[4]))
    if do_passive:
        dataset['composite_change']   = make_composite(dataset, labels=define_composite(num_images=12, changes=[4]))
    return dataset

def define_composite(num_images=9,omissions=[], changes=[]):
    intersection_check = set(omissions).intersection(set(changes))
    if len(intersection_check) > 0:
        raise Exception('Cant have an omission and change at the same time')
    labels = ['i']*num_images
    for omission in omissions:
        labels[omission] = 'o'
    for change in changes:
        labels[change] = 'c'
    return labels

def make_composite(dataset,labels):
    '''
        Generates a composite sequence of normal data.

        images, and omissions are kernel_dictionaries containing the average kernel
        for each cre-line

        labels is a list of strings that indicate what type of image happened at that
        flash. either 'i' for image, 'o' for omission, or 'c' for change 
    '''
    composite={}
    
    # Build time stamps
    time = []
    for i in range(0,len(labels)):
        time.append(0.75*i+dataset['images']['time'])
    composite['time'] = np.concatenate(time)

    # Build each cre-line response
    cre = set(dataset['images'].keys())
    cre.remove('time')
    cre.remove('meta')
    image_indexes = list(range(0,len(composite['time']),len(dataset['images']['time'])))
    for c in cre:
        temp = np.empty(len(composite['time']))
        temp[:] = 0
        # Iterate over each image, and determine what kernels to add
        # all kernels are aligned to image onset
        for dex,index in enumerate(image_indexes):
            if labels[dex] in ['i','c']:
                temp[index:index+len(dataset['images']['time'])]+=dataset['images'][c]
            if labels[dex] == 'o':
                temp[index:index+len(dataset['omissions']['time'])]+=dataset['omissions'][c]           
            if labels[dex] == 'c':
                temp[index:index+len(dataset['changes']['time'])]+=dataset['changes'][c]
        for dex, index in enumerate(image_indexes[1:]):
            temp = clean_alias(temp,3,index) 
        composite[c] = temp
    composite['meta'] = dataset['images']['meta'].copy()
    composite['meta']['kernel'] = 'composite'
    composite['meta']['composite_labels'] = labels
    return composite

def plot_dataset(kernels):
    '''
        Plots the average activity for each cre line in the dataset, and adds stimulus times
    '''
    if kernels['time'][-1] > 6:
        plt.figure(figsize=(10,4))
    else:
        plt.figure()
    cre = set(kernels.keys())
    cre.remove('time')
    cre.remove('meta')
    colors = project_colors()
    for c in cre:
        if 'normalization' not in c:
            plt.plot(kernels['time'],kernels[c],'-', label=c,color=colors[c])   
    plt.legend()
    ax = plt.gca()
    if 'meta' in kernels:
        if kernels['meta']['kernel'] == 'composite':
            add_stimulus_bars(ax,kernels['meta']['kernel'],labels=kernels['meta']['composite_labels'])
        else:
            add_stimulus_bars(ax,kernels['meta']['kernel'])
    ax.axhline(0, color='k',linestyle='--',alpha=0.15)
    ax.axvline(0, color='k',linestyle='--',alpha=0.15)
    if 'meta' in kernels:
        label =''
        if kernels['meta']['data_type'] == 'events':
            label += 'events'
        else:
            label += '$\Delta$f/f'
        ax.set_ylabel(label,fontsize=18)   
    else:
        ax.set_ylabel('Normalized df/f',fontsize=18)   
    ax.set_xlabel('Time (s)',fontsize=18)
    ax.set_xlim(kernels['time'][0],kernels['time'][-1])   
    plt.tick_params(axis='both',labelsize=16)
    plt.title(kernels['meta']['normalization'])
    plt.tight_layout() 


def add_stimulus_bars(ax,kernel,alpha=0.15,labels=[]):
    '''
        Adds stimulus bars to the given axis, but only for certain kernels 
    '''
    # Check if this is an image aligned kernel
    if kernel in ['change','hits','misses','false_alarms','omissions','image_expectation','image0','image1','image2','image3','image4','image5','image6','image7','images','image','composite']:
        # Define timepoints of stimuli
        lims = ax.get_xlim()
        times = set(np.concatenate([np.arange(0,lims[1],0.75),np.arange(-0.75,lims[0]-0.001,-0.75)]))
        change_times = set()
        if kernel == 'composite':
            for dex,label in enumerate(labels):
                if label == 'o':
                    times.remove(0.75*dex)
                if label == 'c':
                    change_times.add(0.75*dex)
                    times.remove(0.75*dex)
        if kernel == 'omissions':
            # For omissions, remove omitted stimuli
            times.remove(0.0)
        if kernel in ['change','hits','misses','false_alarms']:
            # For change aligned kernels, plot the two stimuli different colors
            for flash_start in times:
                if flash_start < 0:
                    ax.axvspan(flash_start,flash_start+0.25,color='green',alpha=alpha,zorder=-np.inf)                   
                else:
                    ax.axvspan(flash_start,flash_start+0.25,color='blue',alpha=alpha,zorder=-np.inf)                   
        else:
            # Normal case, just plot all the same color
            for flash_start in times:
                ax.axvspan(flash_start,flash_start+0.25,color='blue',alpha=alpha,zorder=-np.inf)
            for flash_start in change_times:
                ax.axvspan(flash_start,flash_start+0.25,color='green',alpha=alpha,zorder=-np.inf)
         
def project_colors():
    '''
        Returns a dictionary of RGB values for relevant project attributes
    '''
    colors = {
        'Sst-IRES-Cre':(158/255,218/255,229/255),
        'sst':(158/255,218/255,229/255),
        'Slc17a7-IRES2-Cre':(255/255,152/255,150/255),
        'slc':(255/255,152/255,150/255),
        'Vip-IRES-Cre':(197/255,176/255,213/255),
        'vip':(197/255,176/255,213/255)}
    return colors

def moving_mean(values, window):
    '''
        Returns values smoothed with a window long moving mean
    '''
    weights = np.repeat(1.0, window)/window
    mm = np.convolve(values, weights, 'valid')
    return mm
  
def clean_alias(kernel, window, index):
    '''
        kernel is a timeseries
        window is the length of the smoothing (must be odd)
        index is the numerical index into kernel to center the smoothing around
    '''
    assert np.mod(window,2) == 1, "must be odd"
    a = window*2
    kernel[index-a+np.int((window-1)/2):index+a-np.int((window-1)/2)] = moving_mean(kernel[index-a:index+a], window)
    return kernel
