import warnings
import visual_behavior_glm.src.GLM_fit_tools as gft
import sys
if sys.version_info.minor <= 7:
    cached_property = property
else:
    from functools import cached_property


class GLM(object):
    '''
    GLM class
    inputs: 
        ophys_experiment_id (int): ID of experiment to fit
        version (int): version of code to use
    '''

    def __init__(self, ophys_experiment_id, version=2):
        
        self.version = version
        self.ophys_experiment_id = ophys_experiment_id
        self.oeid = self.ophys_experiment_id
        self.run_params = gft.load_run_json(self.version)
        self.current_model = 'Full'

        self.fit_model()

    def fit_model(self):

        self.session, self.fit, self.design = gft.fit_experiment(
            self.oeid, self.run_params)

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
