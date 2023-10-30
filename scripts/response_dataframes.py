import visual_behavior_glm.build_dataframes as bd

import argparse

parser = argparse.ArgumentParser(description='build response dataframe for experiment')
parser.add_argument(
    '--ophys_experiment_id', 
    type=int, 
    default=0,
    metavar='oeid',
    help='ophys_experiment_id'
)

if __name__ == '__main__':
    args = parser.parse_args()
    session = bd.load_data(args.ophys_experiment_id)
    data = 'events'

    #print('first half') 
    #bd.build_response_df_experiment(session,data,first=True,second=False,image=False)

    #print('second half') 
    #bd.build_response_df_experiment(session,data,first=False,second=True,image=False)
    
    #print('image period')
    #bd.build_response_df_experiment(session,data,first=False,second=False, image=True)

    print('full interval') 
    bd.build_response_df_experiment(session,data,first=False,second=False)

    print('behavior')
    bd.build_behavior_df_experiment(session)
    print('grand finished')
