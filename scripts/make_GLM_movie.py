import argparse
import time
from visual_behavior_glm.glm import GLM
import visual_behavior_glm.GLM_visualization_tools as gvt

parser = argparse.ArgumentParser(description='generate GLM movie')
parser.add_argument(
    '--oeid', 
    type=int, 
    default=0,
    metavar='oeid',
    help='ophys experiment ID'
)

parser.add_argument(
    '--glm-version', 
    type=str, 
    default='',
    metavar='glm_version',
    help='glm version to use'
)

parser.add_argument(
    '--cell-id', 
    type=int, 
    default=0,
    metavar='cell_specimen_id',
    help='cell specimen ID'
)
parser.add_argument(
    '--start-frame', 
    type=int, 
    default=0,
    metavar='frame_0',
    help='first frame of movie'
)
parser.add_argument(
    '--end-frame', 
    type=int, 
    default=0,
    metavar='frame_0',
    help='first frame of movie'
)
parser.add_argument(
    '--frame-interval', 
    type=int, 
    default=1,
    metavar='frame_interval',
    help='step between frames'
)
parser.add_argument(
    '--fps', 
    type=int, 
    default=5,
    metavar='fps',
    help='playback speed'
)

def make_movie(oeid, glm_version, cell_specimen_id, start_frame, end_frame, frame_interval, fps):
    t0=time.time()
    glm = GLM(oeid, version=glm_version, use_previous_fit=True)
    print('fitting the model took {} seconds'.format(time.time()-t0))
    
    movie = gvt.GLM_Movie(
        glm,
        cell_specimen_id = cell_specimen_id, 
        start_frame = start_frame,
        end_frame = end_frame,
        frame_interval = frame_interval,
        fps = fps
    )
    t0=time.time()
    movie.make_movie()
    print('making the movie took {} seconds'.format(time.time()-t0))


if __name__ == '__main__':
    args = parser.parse_args()

    make_movie(
        int(args.oeid), 
        args.glm_version, 
        int(args.cell_id), 
        int(args.start_frame), 
        int(args.end_frame),
        int(args.frame_interval),
        int(args.fps)
    )