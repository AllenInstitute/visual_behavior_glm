import argparse

parser = argparse.ArgumentParser(description='generate all summary figures for a given GLM version')
parser.add_argument(
    '--glm-version', 
    type=int, 
    default=0,
    metavar='glm_version',
    help='version of GLM to use'
)

def generate_all_figures(glm_version):
    results = gat.retrieve_results()

if __name__ == '__main__':
