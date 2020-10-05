#!/bin/bash
echo here
pwd
python temp.py
python make_GLM_movie.py\
    --oeid 953452368\
    --glm-version 7_L2_optimize_by_session\
    --cell-id 1007051748\
    --start-frame 20000\
    --end-frame 22000\
    --frame-interval 10\
    --fps 15\