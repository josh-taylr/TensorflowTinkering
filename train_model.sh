#!/bin/sh
python learn.py --summary_data /tmp/mnist/softmax
tensorboard --logdir=/tmp/mnist/softmax --host 0.0.0.0 --port 6006