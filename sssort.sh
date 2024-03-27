#!/bin/bash
source /home/$USER/anaconda3/etc/profile.d/conda.sh # TODO to determined by setup.py
conda activate sssort
INSTALL_DIR = XX # to be inserted by setup.py
python $INSTALL_DIR/run.py $@
