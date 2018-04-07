#!/usr/bin/env sh
LOG=./resnet27-train-log-`date +%Y-%m-%d-%H:%M:%S`.log
PYDIR=/home/junyang/anaconda2/bin
nohup $PYDIR/python resnet27_main.py --train_list /data/hjy1312/experiments/dagan_combine_uvgan/resnet27_training_list_crop192_144.txt \
 --cuda --ngpu 1 --outf ./train_resnet27_`date +%Y-%m-%d-%H:%M:%S` 2>&1 | tee $LOG&

