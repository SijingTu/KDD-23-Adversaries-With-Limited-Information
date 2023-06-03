#!/bin/bash

python cal_local.py ../data/sbm.pkl sbm 0.1 "NonAdaptiveGreedy"
python cal_local.py ../data/sbm.pkl sbm 0.1 "AdaptiveGreedy"
python cal_local.py ../data/sbm.pkl sbm 0.1 "LocalNonAdaptiveGreedy"
python cal_local.py ../data/sbm.pkl sbm 0.1 "LocalAdaptiveGreedy"
python cal_local.py ../data/sbm.pkl sbm 0.1 "LocalHighDegree"
python cal_local.py ../data/sbm.pkl sbm 0.1 "LocalRandom"
python cal_local.py ../data/sbm.pkl sbm 0.1 "LocalSDP"
python cal_local.py ../data/sbm.pkl sbm 0.1 "LocalIM"