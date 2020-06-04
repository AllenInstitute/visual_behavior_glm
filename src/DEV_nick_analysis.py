##### Evaluation/visualization functions (potentially not up to date) below here #######

def all_cells_psth(dff_traces_arr, ophys_timestamps, flash_time_gb, change_events):

    # Get psth for each flash
    all_psth = {}
    for image_name in flash_time_gb.index.levels[0].values:
        times_this_image = flash_time_gb[image_name].values
        dff_frames_this_image = rp.index_of_nearest_value(ophys_timestamps, times_this_image)
        dff_frames_this_image = dff_frames_this_image[
            (len(ophys_timestamps) - dff_frames_this_image > 31)
        ]
        data_this_image = rp.eventlocked_traces(dff_traces_arr, dff_frames_this_image,
                                                0, 30)
        psth_this_image = data_this_image.mean(axis=1)
        all_psth.update({image_name:psth_this_image})

    # Get psth for average change effect
    dff_frames_changes = rp.index_of_nearest_value(ophys_timestamps, change_events)
    dff_frames_changes = dff_frames_changes[
        (len(ophys_timestamps) - dff_frames_changes > 101)
    ]
    data_changes = rp.eventlocked_traces(dff_traces_arr, dff_frames_changes,
                                            0, 100)
    psth_changes = data_changes.mean(axis=1)
    all_psth.update({'change':psth_changes})
    
    return all_psth

def split_filters(W, image_names):
    '''
    W: n_params, n_cells
    image_names: list of strings
    '''

    start=0
    all_filters = {}
    for ind_image, image_name in enumerate(image_names):
        all_filters.update({image_name:W[start:start+30, :]})
        start += 30
    all_filters.update({'change':W[start:]})
    return all_filters

# TODO what does this function do?
def compare_filter_and_psth(ind_cell, dff_traces_arr, ophys_timestamps, flash_time_gb, change_events, all_W):
    all_psths = all_cells_psth(dff_traces_arr.T, ophys_timestamps, flash_time_gb, change_events)
    image_names = flash_time_gb.index.levels[0].values
    all_filters_mean = split_filters(all_W.mean(axis=2), image_names)
    all_filters_std = split_filters(all_W.std(axis=2), image_names)
    plt.figure()
    num_images = len(image_names)
    for ind_image, image_name in enumerate(image_names):
        plt.subplot(num_images+1, 1, ind_image+1)
        this_psth = all_psths[image_name][:,ind_cell]
        this_filter_mean = all_filters_mean[image_name][:,ind_cell]
        this_filter_std = all_filters_std[image_name][:,ind_cell]
        plt.plot(this_psth, 'k-')
        plt.plot(this_filter_mean, 'r--')
        plt.plot(this_filter_mean+this_filter_std, 'r--', alpha=0.5)
        plt.plot(this_filter_mean-this_filter_std, 'r--', alpha=0.5)
        plt.ylabel(image_name)

    # Average change
    plt.subplot(num_images+1, 1, ind_image+2)
    this_psth = all_psths['change'][:,ind_cell]
    this_filter_mean = all_filters_mean['change'][:,ind_cell]
    this_filter_std = all_filters_std['change'][:,ind_cell]
    plt.plot(this_psth, 'k-')
    plt.plot(this_filter_mean, 'r--')
    plt.plot(this_filter_mean+this_filter_std, 'r--', alpha=0.5)
    plt.plot(this_filter_mean-this_filter_std, 'r--', alpha=0.5)
    plt.ylabel('change')

def compare_all_filters_and_psth(ind_cell, dff_traces_arr, ophys_timestamps, flash_time_gb, change_events, all_W):
    all_psths = all_cells_psth(dff_traces_arr.T, ophys_timestamps, flash_time_gb, change_events)
    image_names = flash_time_gb.index.levels[0].values
    #  all_filters_mean = split_filters(all_W.mean(axis=2), image_names)
    #  all_filters_std = split_filters(all_W.std(axis=2), image_names)
    plt.figure()
    num_images = len(image_names)
    for ind_split in range(6):
        filters_this_split = split_filters(all_W[:, :, ind_split], image_names)
        for ind_image, image_name in enumerate(image_names):
            plt.subplot(num_images+1, 1, ind_image+1)
            this_psth = all_psths[image_name][:,ind_cell]

            this_filter = filters_this_split[image_name][:,ind_cell]
            plt.plot(this_psth, 'k-')
            plt.plot(this_filter, 'r--')
            plt.ylabel(image_name)

def plot_filters(w, image_names):
    num_subplots = len(image_names)+2
    fig, axes = plt.subplots(3, 1)
    start=0
    for ind_image, image_name in enumerate(image_names):
        #  plt.subplot(num_subplots, 1, ind_image+1)
        axes[0].plot(np.linspace(0, 1-1/31, 30), w[start:start+30], label=image_name)
        #  plt.title(image_name)
        #  plt.plot(w[start:start+30])
        start += 30
    axes[0].legend()

    #  plt.subplot(num_subplots, 1, num_subplots)
    axes[1].plot(np.arange(0, 100/31, 1/31), w[start:start+100], label='change')
    start += 100
    axes[1].legend()
    
    axes[2].plot(np.arange(0, 100/31, 1/31), w[start:], label='reward')
    axes[2].legend()

def pref_stim(session, cell_ind):
    csid = session.dff_traces.iloc[cell_ind].name
    stim_id = session.stimulus_response_df.query('cell_specimen_id==@csid and pref_stim').iloc[0]['stimulus_presentations_id']
    return session.stimulus_presentations.loc[stim_id]['image_name']

def fit_and_evaluate(session, ind_cell, X):
    dff_trace, w = fit_cell(session, ind_cell, X)
    plot_filters(w, image_names)
    var_explained = variance_ratio(dff_trace, w, X)
    actual_pref = pref_stim(session, ind_cell)
    plt.suptitle('var explained: {}\npref stim: {}'.format(var_explained, actual_pref))
    plt.tight_layout()

def pref_image_filter(w, image_names):
    start=0
    filter_sums = np.empty(len(image_names))
    for ind_image, image_name in enumerate(image_names):
        filter_sums[ind_image] = np.sum(w[start:start+30])
        start += 30
    return image_names[np.argmax(filter_sums)]

def plot_cv_train_test(cv_train, cv_test, lambda_vals):
    train_mean = cv_train.mean((0, 1))
    test_mean = cv_test.mean((0, 1))
    fig, ax1 = plt.subplots()
    ax1.plot(test_mean, color='r')
    ax1.set_ylabel('cv var explained, test')
    ax1.tick_params(axis='y', labelcolor='r')

    ax2 = ax1.twinx()
    ax2.plot(train_mean, 'k')
    ax2.set_ylabel('cv var explained, train')
    ax2.tick_params(axis='y', labelcolor='k')

    ax1.set_xticks(range(len(lambda_vals)))
    ax1.set_xticklabels(np.round(lambda_vals, 2), rotation='vertical')
    ax1.set_xlabel('lambdas')
