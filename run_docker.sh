#!/bin/sh

docker run -it -v $(pwd):/root -h 0.0.0.0 -p 7007:6006 tensorflow/tensorflow:1.1.0-devel