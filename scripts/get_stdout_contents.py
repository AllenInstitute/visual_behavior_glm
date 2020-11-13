import os
import numpy as np
import pandas as pd
import time
import argparse
from visual_behavior import database as db

def search_file(filepath,search_phrase):
    '''
    search a file for a search_phrase
    return True if search_phrase is found in file, False otherwise
    '''
    searchfile = open(filepath, "r")
    for line in searchfile:
        if search_phrase in line:
            searchfile.close()
            return True
    searchfile.close()
    return False

def get_file_contents(filepath):
    '''
    get all text contents from a file
    '''
    contents = open(filepath, "r")
    text = ''
    for line in contents:
        text += line
    contents.close()
    return text

def get_timestamp(filepath):
    '''
    get PBS job timestamp from a pbs output file
    '''
    text = get_file_contents(filepath)
    date_string  = text.split('End PBS Epilogue ')[1].split('\n')[0]
    day_of_week, month, day, time, tz, year, _ = date_string.split(' ')
    dt = pd.to_datetime('{}-{}-{}T{}'.format(year, month, day,time)).tz_localize('America/Los_Angeles')
    return str(dt)

def get_required_walltime(filepath):
    '''
    get walltime from a pbs output file
    '''
    text = get_file_contents(filepath)
    resources_line = text.split('Resources')[1].split('\n')[0]
    return resources_line.split('walltime=')[1].split(',')[0]

def get_required_memory(filepath, mem_type='mem'):
    '''
    get required memory timestamp from a pbs output file
    must specify virtual memory ('vmem') or memory ('mem')
    '''
    text = get_file_contents(filepath)
    resources_line = text.split('Resources')[1].split('\n')[0]
    return int(resources_line.split(',{}='.format(mem_type))[1].split('kb')[0])

def find_filename_containing_string(search_phrase, search_path, filename_extension, verbose=False):
    '''
    search all files in a directory for a given string
    returns the filename if found, None otherwise
    '''
    files_to_search = [f for f in os.listdir(search_path) if f.endswith(filename_extension)]
    for ii,filename in enumerate(np.sort(files_to_search)[::-1]):
        if verbose:
            print('searching file #{}, name: {}'.format(ii, filename), end='\r')
        if search_file(os.path.join(search_path, filename), search_phrase):
            return filename
            
def search_for_oeid(oeid, glm_version,search_path='/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/cluster_jobs/ophys_glm', verbose=False):
    '''
    search all .out files in a directory for a given oeid/glm_version STDOUT file
    '''
    if verbose:
        print('searching for oeid = {}, glm_version = {}'.format(oeid, glm_version))
    search_phrase = 'oeid_{}_fit_glm_v_{}'.format(oeid, glm_version)
    filename = find_filename_containing_string(search_phrase, search_path, filename_extension='.out', verbose=verbose)
    return os.path.join(search_path, filename)

def build_mongo_stdout_entry(oeid, glm_version):
    '''
    build a mongo entry for the cluster_stdout collection for a given oeid/glm_version
    '''
    filename = search_for_oeid(oeid, glm_version, verbose=True)
    file_text = get_file_contents(filename)
    
    entry = {
        'ophys_experiment_id':oeid,
        'glm_version':glm_version,
        'filename':filename,
        'file_text':file_text,
        'job_timestamp':get_timestamp(filename),
        'required_walltime':get_required_walltime(filename),
        'required_mem':get_required_memory(filename, mem_type='mem'),
        'required_vmem':get_required_memory(filename, mem_type='vmem'),
    }
    return db.clean_and_timestamp(entry)

def log_to_mongo(entry):
    '''
    log entry to mongo
    '''
    conn = db.Database('visual_behavior_data')
    collection = conn['ophys_glm']['cluster_stdout']
    db.update_or_create(collection, entry, keys_to_check = ['ophys_experiment_id','glm_version','job_timestamp'])
    conn.close()

def find_and_log_stdout(oeid, glm_version,verbose=True):
    t0=time.time()
    entry = build_mongo_stdout_entry(oeid, glm_version)
    log_to_mongo(entry)
    if verbose:
        print('\ndone, that took {} seconds'.format(time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='get stdout contents')
    parser.add_argument('--oeid', type=int, default=0, metavar='oeid')
    parser.add_argument('--glm-version', type=str, default='7_L2_optimize_by_session', metavar='glm_version')
    args = parser.parse_args()

    find_and_log_stdout(args.oeid, args.glm_version)