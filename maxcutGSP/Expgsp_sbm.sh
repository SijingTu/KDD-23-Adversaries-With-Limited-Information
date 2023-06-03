#!/bin/bash

python cal_gsp.py ../data/sbm.pkl 0.1 "NonAdaptiveGreedy"
python cal_gsp.py ../data/sbm.pkl 0.1 "AdaptiveGreedy"
python cal_gsp.py ../data/sbm.pkl 0.1 "HighDegree"
python cal_gsp.py ../data/sbm.pkl 0.1 "Random"