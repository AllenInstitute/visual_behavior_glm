# In the dataframe "df", bin a feature (lets call it the primary feature: "parameters_primary"), and then for each bin, plot the distribution of a number of other features (continuous features: "parameters_cont"; categorical features: "parameters_categ")

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_hist_parameter_bins(df, parameter_primary, parameters_cont, parameters_categ, dosavefig_fign=[0,'']):

    ### bin task_dropout_index into 2
    topc = df[parameter_primary].values
    nbins0 = 2

    r = np.max(topc) - np.min(topc)
    binEvery = r/float(nbins0)
    # set bins
    bn = np.arange(np.min(topc), np.max(topc), binEvery)
    # bn = np.concatenate((bn, [np.max(topc)+binEvery/10.])) # doesnt seem like we need this. digitize uses max as the last bin by default
    # bn[-1] = np.max(topc)#+binEvery/10. # unlike digitize, histogram doesn't count the right most value
    # print(bn)
    # print(np.min(topc), np.max(topc))

    hist_inds = np.digitize(topc, bn) # starts at 1!
    hist_vals = [sum(hist_inds==ihi) for ihi in np.unique(hist_inds)]
#     print(np.unique(hist_inds))


    ### plot a histogram of the primary parameter
    plt.figure(figsize=(2,2))
    plt.bar(bn+binEvery/2., hist_vals)
    plt.title(f'{hist_vals} : number of cells per bin of {parameter_primary}')
    plt.xlabel(f'{parameter_primary} (mean per bin)')
    plt.ylabel(f'number of cells')

    
    
    
    #### continuous parameters  
    # now for each bin, look at the distribution of a given parameter

    cols = sns.color_palette("hsv", nbins0) #'b', 'r' # for each bin of the primary parameter

    # nbins0+1; 1: for categorical params
    fig,ax = plt.subplots(2, max(len(parameters_cont), len(parameters_categ)),figsize=(18,8))

    for hind in np.unique(hist_inds): # 1,2,etc. #hind = 1

        sd_this_bin = df[hist_inds == hind]

        for col,parameter in enumerate(parameters_cont):
            # col = 0; parameter = parameters[0] #'imaging_depth' #'all-images'

            ###### bin the secondary parameter
            topc = sd_this_bin[parameter]

            if parameter=='imaging_depth':
                nbins = 25
            else:        
                nbins = 200

            r = np.max(topc) - np.min(topc)
            binEvery = r/float(nbins)
            # set bins
            bn = np.arange(np.min(topc), np.max(topc), binEvery)
            bn = np.concatenate((bn, [np.max(topc)+binEvery/10.]))
            ######

            row = hind - 1
            row_ax = 0

            ax[row_ax,col].hist(
                sd_this_bin[parameter],
                bins=bn,
                density=True,
                color=cols[row],
                alpha=.35,
                label=f'bin {row}'
            );


            ax[row_ax,col].set_xlim([np.min(df[parameter]), np.max(df[parameter])])
            
            # set titles, make first row title different
            ax[row_ax, col].set_title('parameter = {}\n{}'.format(parameter, parameter_primary))
            '''
            if row_ax == 0:
                ax[row, col].set_title('parameter = {}\nbin {}, {}'.format(parameter, row, parameter_primary))
            else:
                ax[row, col].set_title('bin {}, {}'.format(row, parameter_primary))
            '''
            ax[row_ax,col].legend()

            if col==0:
                ax[row_ax,col].set_ylabel('Fraction cells')

                
                
                
    ############ categorical parameters ############   
    # now for each bin, look at the pie chart of a given parameter

    # fig, ax = plt.subplots(1, len(parameters),figsize=(18,4))
    for hind in np.unique(hist_inds): # 1,2,etc. #hind = 1

        sd_this_bin = df[hist_inds == hind]

        for col,parameter in enumerate(parameters_categ):
            # col = 0; parameter = parameters[0]

            row = hind - 1

            topc = sd_this_bin[parameter].value_counts()

            x0 = np.arange(len(topc))
            x = x0 + row*.15
            y = topc.values / sum(topc.values)
            
            row_ax = 1 # 2
            
            ax[row_ax,col].bar(x, y, color=cols[row], alpha=.5, width=.1, label=f'bin {row}')

            xlabs = (topc.index.values).astype(str)
            if parameter=='session_id':
                xlabs[xlabs=='0'] = 'Familiar'
                xlabs[xlabs=='1'] = 'Novel'

            if parameter=='cre_line':
                xlabs = [xl[:3] for xl in xlabs]

            ax[row_ax,col].set_xticks(x0)
            ax[row_ax,col].set_xticklabels(xlabs)
            ax[row_ax,col].set_title('parameter = {}'.format(parameter))

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

    plt.subplots_adjust(hspace=.5)

    
    
    ### save the figure
    
    dosavefig = dosavefig_fign[0]
    fign = dosavefig_fign[1]
    
    if dosavefig:
#         nam = f'bins_{parameter_primary}_param_hists_allCre_{now}'
#         fign = os.path.join(dir0, dir_now, nam+fmt)     

        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    
          