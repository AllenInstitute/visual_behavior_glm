# In the dataframe "df", bin a feature (lets call it the primary feature: "parameters_primary"), and then for each bin, plot the distribution of a number of other features (continuous features: "parameters_cont"; categorical features: "parameters_categ")

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_hist_parameter_bins(df, parameter_primary, parameters_cont, parameters_categ, norm_by_param_primary=0, dosavefig_fign=[0,'']):

    # norm_by_param_primary = 0 # if 1: # normalize each value of parameters_categ (that is in a given bin of param_primary) by the total number of cells in that bin of param_primary (eg normalize V1 neurons in bin0 of task_dropout_index by the sum of V1 and LM neurons in bin0 of task_dropout_index ).  If 0, # normalize each value of parameters_categ (that is in a given bin of param_primary) by the total number of cells that have that given value of param_categ (acorss all bins of param_primary); (eg normalize V1 neurons in bin0 of task_dropout_index by the total number of V1 neurons (ie in all bins of task_dropout_index)).
    
    ### bin task_dropout_index into 2
    topc = df[parameter_primary].values
    nbins0 = 2

    r = np.max(topc) - np.min(topc)
#     binEvery = r/float(nbins0)
    # set bins
#     bn = np.arange(np.min(topc), np.max(topc), binEvery)
    bn0 = [np.min(topc), np.median(topc)] #, np.max(topc)]
    binEvery0 = np.diff(np.array([np.min(topc), np.median(topc), np.max(topc)]))
    
    # bn = np.concatenate((bn, [np.max(topc)+binEvery/10.])) # doesnt seem like we need this. digitize uses max as the last bin by default
    # bn[-1] = np.max(topc)#+binEvery/10. # unlike digitize, histogram doesn't count the right most value
    # print(bn)
    # print(np.min(topc), np.max(topc))

    hist_inds = np.digitize(topc, bn0) # starts at 1!
    hist_vals = [sum(hist_inds==ihi) for ihi in np.unique(hist_inds)]
#     print(np.unique(hist_inds))

    hist_bins_mean = [np.mean(topc[hist_inds==ibins]) for ibins in np.unique(hist_inds)]
    hist_bins_mean = np.round(hist_bins_mean, 2)
    
    
    ################################################
    ############ continuous parameters ############# 
    ################################################
    # now for each bin, look at the distribution of a given parameter

    cols = sns.color_palette("hsv", nbins0) #'b', 'r' # for each bin of the primary parameter
    nrows_cont = 1 #int(np.ceil(len(parameters_cont)/5))
    
    # nbins0+1; 1: for categorical params
    fig,ax = plt.subplots(nrows_cont+1, max(len(parameters_cont), len(parameters_categ)),figsize=(20,8))

    for hind in np.unique(hist_inds): # 1,2,etc. #hind = 1

        sd_this_bin = df[hist_inds == hind]

        for col,parameter in enumerate(parameters_cont):
            # col = 0; parameter = parameters_cont[col] #'imaging_depth' #'all-images'

            ###### bin the secondary parameter
            topc = sd_this_bin[parameter]

            if parameter=='imaging_depth':
                nbins = 25
            else:        
                nbins = 20 #200
            
            topcc = df[parameter] # use the same set of param_cont bins for both bins of param_primary; eg for all-images that are in bin0 of task_dropout_index and those that are in its bin1, use the same set of bins.
#             topcc = topc
            r = np.max(topcc) - np.min(topcc)
            binEvery = r/float(nbins)
            # set bins
            bn = np.arange(np.min(topcc), np.max(topcc), binEvery)
            bn = np.concatenate((bn, [np.max(topcc)+binEvery/10.]))
            

            ###### compute histogram
            hist, bin_edges = np.histogram(topc, bins=bn) # bins='sturges') #
#             print(bin_edges.shape)
#             print(bn)
            
            if ~np.isnan(norm_by_param_primary): # even when it is 0, we normalize it by the sum within a given bin of primary parameter
                hist = hist/float(np.sum(hist))
                

            ###### plot the histogram
            row = hind - 1
            row_ax = 0 # max(0, int(np.ceil(col/5)-1)) # 0
#             print(col, row_ax)
            
            # plot the center of bins
            ax[row_ax,col].bar(bin_edges[0:-1]+binEvery/2., hist, color=cols[row], label=f'{hist_bins_mean[row]}', alpha=.35, width=r/nbins) 

            '''
            ax[row_ax,col].hist(
                sd_this_bin[parameter],
                bins=bn,
                density=True,
                color=cols[row],
                alpha=.35,
                label=f'{hist_bins_mean[row]}'
            );
            '''
            
            mnx = (np.max(df[parameter]) - np.min(df[parameter]))/10
            ax[row_ax,col].set_xlim([np.min(df[parameter])-mnx, np.max(df[parameter])+mnx])
            
            ax[row_ax, col].set_xlabel(f'{parameter}')
#             ax[row_ax, col].set_title(f'bins of {parameter_primary}')
            '''
            if row_ax == 0:
                ax[row, col].set_title('parameter = {}\nbin {}, {}'.format(parameter, row, parameter_primary))
            else:
                ax[row, col].set_title('bin {}, {}'.format(row, parameter_primary))
            '''
            ax[row_ax,col].legend()

            if col==0:
                ax[row_ax,col].set_ylabel('Fraction cells')

                
                
    ################################################            
    ############ categorical parameters ############   
    ################################################
    # now for each bin, look at the pie chart of a given parameter

    # fig, ax = plt.subplots(1, len(parameters),figsize=(18,4))
    for hind in np.unique(hist_inds): # 1,2,etc. #hind = 1

        sd_this_bin = df[hist_inds == hind]

        for col,parameter in enumerate(parameters_categ):
            # col = 3; parameter = parameters_categ[col]
#             print(hind,parameter)
            
            row = hind - 1

            topc = sd_this_bin[parameter].value_counts() # number of neurons with different values of a given param_categ (all in a given bin of param_primary); (eg number of V1 and LM neurons (when param_categ is targeted_structure); all of these neurons have bin0 value of param_primary

            y0 = topc #.values 
            
            if norm_by_param_primary==1: # if 1: # normalize each value of parameters_categ (that is in a given bin of param_primary) by the total number of cells in that bin of param_primary (eg normalize V1 neurons in bin0 of task_dropout_index by the sum of V1 and LM neurons in bin0 of task_dropout_index ).  If 0, # normalize each value of parameters_categ (that is in a given bin of param_primary) by the total number of cells that have that given value of param_categ (acorss all bins of param_primary); (eg normalize V1 neurons in bin0 of task_dropout_index by the total number of V1 neurons (ie in all bins of task_dropout_index)).
                # in the barplots, same color bars will sum to 1.
                y = y0 / sum(topc.values) # normalize each value of parameters_categ (that is in a given bin of param_primary) by the total number of cells in that bin of param_primary (eg normalize V1 neurons in bin0 of task_dropout_index by the sum of V1 and LM neurons in bin0 of task_dropout_index ).
            elif norm_by_param_primary==0:
                # in the barplots, different colors of bars (of a given value of param_categ) will sum to 1.
                y = y0 / df[parameter].value_counts() # normalize each value of parameters_categ (that is in a given bin of param_primary) by the total number of cells that have that given value of param_categ (acorss all bins of param_primary); (eg normalize V1 neurons in bin0 of task_dropout_index by the total number of V1 neurons (ie in all bins of task_dropout_index)).
            elif np.isnan(norm_by_param_primary): # dont normalize
                y = y0
                
            x0 = np.arange(len(y))
            x = x0 + row*.15
            xlabs = (y.index.values).astype(str) #(topc.index.values).astype(str)
            
            row_ax = 1 #max(np.ceil((0+len(parameters_cont))/5), int(np.ceil((col+len(parameters_cont))/5)-1)) # 1 # 2
#             print(col, row_ax)
            
            ax[row_ax,col].bar(x, y, color=cols[row], alpha=.5, width=.08, label=f'{hist_bins_mean[row]}')

            if parameter=='session_id':
                xlabs[xlabs=='0'] = 'Familiar'
                xlabs[xlabs=='1'] = 'Novel'

            if parameter=='cre_line':
                xlabs = [xl[:3] for xl in xlabs]
#             print(x0, xlabs)
            ax[row_ax,col].set_xticks(x0)
            ax[row_ax,col].set_xticklabels(xlabs, rotation=45, fontsize=11)
            ax[row_ax, col].set_xlabel(f'{parameter}')
#             ax[row_ax, col].set_title(f'bins of {parameter_primary}')
            
#             ax[row_ax, col].set_title('parameter = {}\n{}'.format(parameter, parameter_primary))
            
            ax[row_ax,col].legend()

            if col==0:
                ax[row_ax,col].set_ylabel('Fraction cells')

            '''
            # build a dataframe of value counts for the pie chart (there's probably a better way!)
            df0 = pd.DataFrame(sd_this_bin[parameter].value_counts()).sort_index()

            plot = df0.plot.pie(
                y = parameter, 
                ax = ax[row,col],
                legend = False
            )        
            ax[row,col].set_ylabel('')
            '''


    ### plot a histogram of the primary parameter
#     plt.figure(figsize=(2,2))
#     print(bn0)
    ax[row_ax, -1].bar(bn0 + binEvery0/2., hist_vals, width=.05)
#     ax[row_ax, -1].set_title(f'{hist_vals} : number of cells per bin of {parameter_primary}')
    ax[row_ax, -1].set_xlabel(f'{parameter_primary} (bins)')
    ax[row_ax, -1].set_ylabel(f'number of cells')

    
    plt.subplots_adjust(hspace=.4, wspace=.4)
    plt.suptitle(f'bins of {parameter_primary}', y=1)
    
    
    ### save the figure
    
    dosavefig = dosavefig_fign[0]
    fign = dosavefig_fign[1]
    
    if dosavefig:
#         nam = f'bins_{parameter_primary}_param_hists_allCre_{now}'
#         fign = os.path.join(dir0, dir_now, nam+fmt)     

        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    
          