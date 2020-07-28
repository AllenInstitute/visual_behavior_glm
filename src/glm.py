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


class GLM(object):
    '''
    GLM class
    inputs: 
        ophys_experiment_id (int): ID of experiment to fit
        version (int): version of code to use
    '''

    def __init__(self, ophys_experiment_id, version):
        
        self.version = version
        self.ophys_experiment_id = ophys_experiment_id
        self.oeid = self.ophys_experiment_id
        self.run_params = glm_params.load_run_json(self.version)
        self.kernels = self.run_params['kernels']
        self.current_model = 'Full'

        self._import_glm_fit_tools()

        self.fit_model()
        self.collect_results()
        self.timestamps = self.fit['dff_trace_arr']['dff_trace_timestamps'].values

    def _import_glm_fit_tools(self):
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

        self.session, self.fit, self.design = self.gft.fit_experiment(
            self.oeid, self.run_params)

    def collect_results(self):
        self.results = self.gft.build_dataframe_from_dropouts(self.fit)
        self.dropout_summary = gat.generate_results_summary(self)

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
        return self.fit['dropouts'][self.current_model]['weights']
