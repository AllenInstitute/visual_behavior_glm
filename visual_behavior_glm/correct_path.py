from pathlib import PurePath
import os


def path(filepath):
    '''
    Corrects path depending if the code is run on Windows or Linux.
    INPUT:
    filepath        path string to fix

    RETURNS:
    corrected_path  with '/' prefix is run on Windows.
    '''
    filepath = PurePath(filepath)  # make sure all are forward slashes
    if os.name == 'nt':
        path_prefix = '/'
    else:
        path_prefix = ''
    corrected_path = path_prefix+filepath
    return corrected_path
