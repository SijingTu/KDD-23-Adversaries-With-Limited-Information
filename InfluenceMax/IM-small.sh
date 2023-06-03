#!/bin/bash

for ratio in 0.1 ; do
    for var in sbm; do
        julia -O3 NetProcess.jl $var $ratio
    done
done

