#!/bin/bash

export PYTHONMALLOC=malloc

valgrind --tool=memcheck \
         --dsymutil=yes \
         --track-origins=yes \
         --show-leak-kinds=all \
         --trace-children=yes \
         python trexio_run.py check-basis ./h2.ezfio.h5
