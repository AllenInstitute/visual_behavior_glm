import warnings
import visual_behavior_glm_strategy.GLM_analysis_tools as gat
import visual_behavior_glm_strategy.GLM_fit_tools as gft
import visual_behavior_glm_strategy.GLM_visualization_tools as gvt
import visual_behavior_glm_strategy.GLM_params as glm_params
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
import visual_behavior.database as db
import numpy as np

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

    def __init__(self, ophys_experiment_id, version, log_results=True, log_weights=True,use_previous_fit=False, 
                recompute=True, use_inputs=False, inputs=None, NO_DROPOUTS=False, TESTING=False):
        
        self.version = version
        self.ophys_experiment_id = ophys_experiment_id
        self.ophys_session_id = db.lims_query('select ophys_session_id from ophys_experiments where id = {}'.format(self.ophys_experiment_id))
        self.oeid = self.ophys_experiment_id
        self.run_params = glm_params.load_run_json(self.version)
        self.kernels = self.run_params['kernels']
        self.current_model = 'Full'
        self.NO_DROPOUTS=NO_DROPOUTS
        self.TESTING=TESTING

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

        try:
            self.timestamps = self.fit['fit_trace_arr']['fit_trace_timestamps'].values
        except KeyError:
            # older versions of model don't have `fit_trace_arr` or `events_trace_arr`
            # in these older versions, the fit trace was always the dff trace
            # fill in missing keys in this case so that code below will run
            self.timestamps = self.fit['dff_trace_arr']['dff_trace_timestamps'].values
            self.fit['fit_trace_arr'] = self.fit['dff_trace_arr'].copy().rename({'dff_trace_timestamps':'fit_trace_timestamps'})
            self.fit['events_trace_arr'] = self.fit['dff_trace_arr'].copy().rename({'dff_trace_timestamps':'fit_trace_timestamps'})
            self.fit['dff_trace_arr'] = self.fit['dff_trace_arr'].rename({'dff_trace_timestamps':'fit_trace_timestamps'})
            # fill events xarray with filtered events values from session
            for idx in range(np.shape(self.fit['events_trace_arr'])[1]):
                csid = int(self.fit['events_trace_arr']['cell_specimen_id'][idx])
                all_events = self.session.events.loc[csid]['filtered_events']
                # only include events during task (excluding gray screens at beginning/end)
                first_index = np.where(self.session.ophys_timestamps >= self.timestamps[0])[0][0]
                self.fit['events_trace_arr'][:,idx] = all_events[first_index:first_index + len(self.timestamps)]

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
            self.oeid, self.run_params, self.NO_DROPOUTS, self.TESTING)

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
        self.results = self.gft.build_dataframe_from_dropouts(self.fit,self.run_params)
        self.dropout_summary = gat.generate_results_summary(self)

        # add roi_ids
        self.dropout_summary = self.dropout_summary.merge(
            self.session.cell_specimen_table[['cell_roi_id']],
            left_on='cell_specimen_id',
            right_index=True,
            how='left'
        )

        self.results = self.results.merge(
            self.session.cell_specimen_table[['cell_roi_id']],
            left_index=True,
            right_index=True,
            how='left'
        )
 
    def get_cells_above_threshold(self, threshold=None): 
        '''
            Returns a list of cells whose full model variance explained is above some threshold
        '''
        if threshold is None:
            if 'dropout_threshold' in self.run_params:
                threshold = self.run_params['dropout_threshold']
            else:
                threshold = 0.005
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
    def dropout_models(self):
        '''
        returns a list of dropout models
        these are the models that were fit with a limited subset of regressors
        '''
        return list(self.fit['dropouts'].keys())
    

    def load_alternate_model(self, desired_model):
        '''
        by default, the glm object contains the 'Full' model
        Calling this method will load the appropriate alternate model
        and will update the following class properties:
            * current_model
            * cell_results_df
            * X
            * W

        Function returns nothing, but the above cached properties will be updated after this function call

        Parameters:
        -----------
        desired_model : str
            desired model to load. Must be among list of possible dropout models
            call self.dropout_models to see a full list of possible models
        
        Returns:
        --------
        None
        '''
        assert desired_model in self.dropout_models, 'desired model must be an existing dropout model: {}'.format(self.dropout_models)

        self.current_model = desired_model

        # Iterate through cached properties and delete them if they exist
        # (keys will not exist if the cached property has not yet been called)
        # This will force them to repopulate on the next time they are called
        for property in ['cell_results_df', 'X', 'W']:
            try:
                del self.__dict__[property]
            except KeyError:
                pass

    @cached_property
    def cell_results_df(self):
        '''creates a tidy dataframe with columns:
            'fit_trace_timestamps': The timestamps of the fit array. Timestamps from ophys sessions
            'frame_index': Index of each frame in the ophys timestamps array. Note that we are trimming off timestamps that occur before the task starts, so this will not start at 0
            'cell_specimen_id': The ID of the cell
            'dff': the delta_F/F signal extracted from the 2p movie - one possible array that can be fit by the model
            'events': Discrete events are derived from dff, then convolved with a half-gaussian filter to give a continuous signal - one possible array that can be fit by the model
            'fit_arr': The array that the model will try to fit. Can be either 'dff' or 'events' (this column will be a duplicate of one of those two columns)
            'model_prediction': The output of the model - Should be similar to 'fit_array' (better model performance = more similar)
        '''

        # build a dataframe with columns for 'fit_array', 'dff_trace_arr', 'events_trace_arr'
        fit_df = self.fit['fit_trace_arr'].to_dataframe(name='fit_array')
        dff_df = self.fit['dff_trace_arr'].to_dataframe(name='dff')
        try:
            event_df = self.fit['events_trace_arr'].to_dataframe(name='events')
        except AttributeError:
            timestamps_to_use = gft.get_ophys_frames_to_use(self.session)
            events_trace_arr = gft.get_events_arr(self.session, timestamps_to_use) 
            event_df = events_trace_arr.to_dataframe(name='events')
        df = fit_df.reset_index().merge(
            dff_df.reset_index(),
            left_on=['fit_trace_timestamps','cell_specimen_id'],
            right_on=['fit_trace_timestamps','cell_specimen_id'],
        )
        
        df = df.merge(
            event_df.reset_index(),
            left_on=['fit_trace_timestamps','cell_specimen_id'],
            right_on=['fit_trace_timestamps','cell_specimen_id'],
        )

        # calculate the prediction matrix Y
        xrt = self.fit['fit_trace_arr'].copy()
        xrt.values = self.X @ self.W

        # add the predictions to the dataframe
        df = df.merge(
            xrt.to_dataframe(name='model_prediction'),
            left_on=['fit_trace_timestamps', 'cell_specimen_id'],
            right_on=['fit_trace_timestamps', 'cell_specimen_id'],
        ).reset_index()

        # add time
        time_df = (
            self
            .fit['fit_trace_arr']['fit_trace_timestamps']
            .to_dataframe()
            .reset_index(drop=True)
            .reset_index()
            .rename(columns = {'index':'frame_index'})
        )

        # merge in time
        df = df.merge(
            time_df,
            left_on = 'fit_trace_timestamps',
            right_on = 'fit_trace_timestamps',
        )

        # adjust frame indices to account for frames that may have been trimmed from start of movie
        first_frame = np.where(self.session.ophys_timestamps >= df.fit_trace_timestamps.min())[0][0]
        df['frame_index'] += first_frame

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

    def make_movie(self, cell_specimen_id, action='make_movie', start_frame=0, end_frame=None, frame_interval=1, fps=10, destination_folder=None, verbose=False):
        '''
        generate a movie to visualize the contribution of various regressors to the model prediction
        inputs:
            cell_specimen_id: the cell to visualize
            action: 'make_movie' or 'display_first_frame'
                make_movie will make the full movie from start_frame to end_frame
                display_first_frame will display a static plot of just start_frame
        '''
        if end_frame is None:
            end_frame = len(self.timestamps)-1

        glm_movie = gvt.GLM_Movie(
            self, 
            cell_specimen_id=cell_specimen_id, 
            start_frame=start_frame, 
            end_frame=end_frame, 
            frame_interval=frame_interval, 
            fps=fps, 
            destination_folder=destination_folder, 
            verbose=verbose
        )
        if action == 'make_movie':
            glm_movie.make_movie()
        elif action == 'display_first_frame':
            glm_movie.make_cell_movie_frame(glm_movie.ax, glm_movie.glm, start_frame, cell_specimen_id)

        return glm_movie

