#!/bin/bash
source /home/$USER/anaconda3/etc/profile.d/conda.sh
conda activate sssort

# 2do put install install path here
python /home/georg/code/SSSort/run.py $@
