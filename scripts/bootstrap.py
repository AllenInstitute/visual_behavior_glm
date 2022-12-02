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
    '--depth', 
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
    '--splits',
    '--list',
    nargs='*',
    help='splits',
    default=[]
    )

parser.add_argument(
    '--query',
    type=str,
    default = ''
    )
parser.add_argument(
    '--extra',
    type=str,
    default = ''
    )

if __name__ == '__main__':
    args = parser.parse_args()
    print('Starting bootstrap with the following inputs')
    print('cell_type {}'.format(args.cell_type))
    print('response  {}'.format(args.response))
    print('data      {}'.format(args.data))
    print('depth     {}'.format(args.depth))
    print('nboots    {}'.format(args.nboots))
    print('splits    {}'.format(args.splits))
    print('query     {}'.format(args.query))
    print('extra     {}'.format(args.extra))
    print('')
    summary_df = po.get_ophys_summary_table(21)
    hierarchy = psth.load_df_and_compute_hierarchy(
        summary_df,
        args.cell_type,
        args.response,
        args.data,
        args.depth,
        args.nboots,
        args.splits,
        args.query,
        args.extra
        )
    print('finished') 
