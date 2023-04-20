import visual_behavior_glm.PSTH as psth
import psy_output_tools as po
import argparse

parser = argparse.ArgumentParser(description='compute hierarchy bootstraps')
parser.add_argument(
    '--cell_type', 
    type=str, 
    default='',
    metavar='cell',
    help='cell_type'
)
parser.add_argument(
    '--response', 
    type=str, 
    default='',
    metavar='response',
    help='response'
)

parser.add_argument(
    '--data', 
    type=str, 
    default='',
    metavar='',
    help='data'
)

parser.add_argument(
    '--nboots', 
    type=int, 
    default=0,
    metavar='',
    help='data'
)

parser.add_argument(
    '--bin_num', 
    type=int, 
    default=0,
    metavar='',
    help='data'
)


if __name__ == '__main__':
    args = parser.parse_args()
    print('Starting bootstrap with the following inputs')
    print('cell_type {}'.format(args.cell_type))
    print('response  {}'.format(args.response))
    print('data      {}'.format(args.data))
    print('nboots    {}'.format(args.nboots))
    print('bin_num   {}'.format(args.bin_num))
    print('')
    summary_df = po.get_ophys_summary_table(21)
    psth.load_df_and_compute_running(
        summary_df,
        args.cell_type,
        args.response,
        args.data,
        args.nboots,
        args.bin_num,
        meso=True
        )
    print('finished') 



