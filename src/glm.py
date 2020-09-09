import warnings
import visual_behavior_glm.src.GLM_analysis_tools as gat
import visual_behavior_glm.src.GLM_visualization_tools as gvt
import visual_behavior_glm.src.GLM_params as glm_params
import sys
import matplotlib.pyplot as plt
if sys.version_info.minor <= 7:
    cached_property = property
else:
    from functools import cached_property
import importlib.util
import sys
import os
import pandas as pd

class GLM(object):
    '''
    GLM class
    inputs: 
        ophys_experiment_id (int): ID of experiment to fit
        version (int): version of code to use
       
        log_results (bool): if True, logs results to mongoDB
        log_weights (bool): if True, logs weights to mongoDB 
        use_previous_fit (bool): if True, attempts to load existing results instead of fitting the model
        recompute (bool): if True, if the attempt to load the existing results fails, will fit the model instead of crashing
        use_inputs (bool): if True, gets session, fit, and design objects from inputs=[session, fit, design]
        inputs (List): if use_inputs, this must be a list of session, fit, and design objects
    '''

    def __init__(self, ophys_experiment_id, version, log_results=True, log_weights=True,use_previous_fit=False, recompute=True, use_inputs=False, inputs=None):
        
        self.version = version
        self.ophys_experiment_id = ophys_experiment_id
        self.oeid = self.ophys_experiment_id
        self.run_params = glm_params.load_run_json(self.version)
        self.kernels = self.run_params['kernels']
        self.current_model = 'Full'  #TODO, what does this do?

        # Import the version's codebase
        self._import_glm_fit_tools()

        if use_inputs & (inputs is not None):
            # If user supplied session, fit, and design objects we dont need to load from file or fit model
            self.session = inputs[0]
            self.fit = inputs[1]
            self.design = inputs[2]
        elif use_previous_fit:
            # Attempts to load existing results
            try:
                print('loading previous fit...')
                self.load_fit_model()  
                print('done loading previous fit')     
            except:
                print('failed to load previous fit, reload flag is set to {}'.format(recompute))
                if recompute:
                    # Just computes the model, since it crashed on load
                    self.fit_model()
                else:
                    raise Exception('Crash during load_fit_model(), check if file exists') 
        else:
            # Fit the model, can be slow
            self.fit_model()
        
        print('done fitting model, collecting results')
        self.collect_results()
        print('done collecting results')
        self.timestamps = self.fit['dff_trace_arr']['dff_trace_timestamps'].values
        if log_results:
            print('logging results to mongo')
            gat.log_results_to_mongo(self) 
            print('done logging results to mongo')
        if log_weights:
            print('logging W matrix to mongo')
            gat.log_weights_matrix_to_mongo(self)
            print('done logging W matrix to mongo')
        print('done building GLM object')

    def _import_glm_fit_tools(self):
        # TODO, need more documentation here
        # we only know the path for loading GLM_fit_tools after loading the run_params
        # therefore, we have to import here, and set the module as an attribute
        import_dir = self.run_params['model_freeze_dir'].rstrip('/')
        module_name = 'GLM_fit_tools'
        file_path = os.path.join(import_dir, module_name+'.py')
        print('importing {} from {}'.format(module_name, file_path))

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        gft = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = gft
        spec.loader.exec_module(gft)
        self.gft = gft

    def fit_model(self):
        '''
        Fits the model
        '''
        self.session, self.fit, self.design = self.gft.fit_experiment(
            self.oeid, self.run_params)

    def load_fit_model(self):
        '''
            Loads existing results. Will crash if file doesn't exist
        '''
        self.session, self.fit, self.design = self.gft.load_fit_experiment(
            self.oeid, self.run_params)

    def collect_results(self):
        '''
            Organizes dropout and model selection results, and adds three dataframes to the object
            self.results is a dataframe of every model-dropout, and information about it.
            self.dropout_summary is the original dropout summary doug implemented, using the non-adjusted variance explained/dropout
            self.adj_dropout_summary uses the adjusted dropout and variance explained 
        '''
        self.results = self.gft.build_dataframe_from_dropouts(self.fit)
        dropout_summary = gat.generate_results_summary(self)
        adj_dropout_summary = gat.generate_results_summary_adj(self)
        self.dropout_summary = pd.merge(dropout_summary, adj_dropout_summary,on=['dropout', 'cell_specimen_id']).reset_index()
        self.dropout_summary.columns.name = None
 
    def get_cells_above_threshold(self, threshold=0.01):
        '''
            Returns a list of cells whose full model variance explained is above some threshold
        '''
        return self.dropout_summary.query('dropout=="Full" & variance_explained > @threshold')['cell_specimen_id'].unique()

    def plot_dropout_summary(self, cell_specimen_id, ax=None):
        '''
        makes a 1x3 matrix of plots showing:
            * Variance explained for each dropout model
            * Absolute change in variance explained for each dropout model relative to the Full model
            * Fraction change in variance explained for each dropout model relative to the Full model
        inputs:
            cell_specimen_id (int): the cell to analyze
            ax (1x3 or 3x1 array or list of matplotlib axes): axes on which to plot. If not passed, new axes will be created
        '''
        if ax is None:
            fig,ax = plt.subplots(1,3,figsize=(15,5))
        gvt.plot_dropout_summary(self.dropout_summary, cell_specimen_id, ax)

    def plot_filters(self, cell_specimen_id, n_cols=5):
        '''plots all filters for a given cell'''
        gvt.plot_filters(self, cell_specimen_id, n_cols)

    @cached_property
    def df_full(self):
        '''creates a tidy dataframe with columns ['dff_trace_timestamps', 'frame_index', 'cell_specimen_id', 'dff', 'dff_predicted] using the full model'''
        df = self.fit['dff_trace_arr'].to_dataframe(name='dff')

        xrt = self.fit['dff_trace_arr'].copy()
        xrt.values = self.X @ self.W

        df = df.merge(
            xrt.to_dataframe(name='dff_predicted'),
            left_on=['dff_trace_timestamps', 'cell_specimen_id'],
            right_on=['dff_trace_timestamps', 'cell_specimen_id'],
        ).reset_index()

        time_df = (
            self
            .fit['dff_trace_arr']['dff_trace_timestamps']
            .to_dataframe()
            .reset_index(drop=True)
            .reset_index()
            .rename(columns = {'index':'frame_index'})
        )

        df = df.merge(
            time_df,
            left_on = 'dff_trace_timestamps',
            right_on = 'dff_trace_timestamps',
        )

        return df

    @cached_property
    def df_dropout(self):
        # TODO: add a long form (aka 'tidy') dataframe for every dropout condition
        warnings.warn('this property is not yet implemented')
        pass

    @property
    def X(self):
        return self.design.get_X(kernels=self.fit['dropouts'][self.current_model]['kernels'])

    @property
    def W(self):
        if 'train_weights' in self.fit['dropouts'][self.current_model].keys():
            return self.fit['dropouts'][self.current_model]['train_weights']
        elif 'weights' in self.fit['dropouts'][self.current_model].keys():
            # to retain backward compatibility prior to merging PR #86
            return self.fit['dropouts'][self.current_model]['weights']
        else:
            warnings.warn('could not locate weights array')
            return None
