# Adversaries with Limited Information in the Friedkin–Johnsen Model

This is a code repository for the SIGKDD'23 accepted paper *Adversaries with Limited Information in the Friedkin–Johnsen Model*.

## Table of Contents

- [Adversaries with Limited Information in the Friedkin–Johnsen Model](#adversaries-with-limited-information-in-the-friedkinjohnsen-model)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
  - [License](#license)

## Installation

The project is written in `Python` and `Julia`, please make sure the relavant packages are installed. 

In particular, the Semidefinite Programming is solved by `Mosek`, so please make sure Mosek, and mosek.fusion are installed in the system. 


## Usage

Here I give a concrete example to run our experiments on a Stochastic Block Model.

Step one: Inside folder `data_preprocess`, run:

```
python create_small_indices.py 2
```

to generate the correct data structure.
After this step, `sbm.pkl` will be created under folder `data`, and `sbm.txt` will be created under folder `InfluenceMax/.in/`.

Step two: Inside folder `InfluenceMax`, run:

```
julia -O3 NetProcess sbm 0.1
```

to find the influential nodes under independent cascade model. To fetch the nodes, run:

```
python fetch_nodes.py 'sbm.feather' 'sbm.pkl' 'sbm' 0.1
```

Step three: Inside folder `maxcutGSP`, run the following codes to find the influential nodes under the limited information model:
```
python cal_gsp.py ../data/sbm.pkl 0.1 "SDP"
python cal_gsp.py ../data/sbm.pkl 0.1 "NonAdaptiveGreedy"
python cal_gsp.py ../data/sbm.pkl 0.1 "AdaptiveGreedy"
python cal_gsp.py ../data/sbm.pkl 0.1 "HighDegree"
python cal_gsp.py ../data/sbm.pkl 0.1 "Random"
```

Step four: Inside folder `maxcutLocal`, run the following nodes to evaluate the nodes found by the limited-information model as well as the full informaiton model:
```
python cal_local.py ../data/sbm.pkl sbm 0.1 "NonAdaptiveGreedy"
python cal_local.py ../data/sbm.pkl sbm 0.1 "AdaptiveGreedy"
python cal_local.py ../data/sbm.pkl sbm 0.1 "LocalNonAdaptiveGreedy"
python cal_local.py ../data/sbm.pkl sbm 0.1 "LocalAdaptiveGreedy"
python cal_local.py ../data/sbm.pkl sbm 0.1 "LocalHighDegree"
python cal_local.py ../data/sbm.pkl sbm 0.1 "LocalRandom"
python cal_local.py ../data/sbm.pkl sbm 0.1 "LocalSDP"
python cal_local.py ../data/sbm.pkl sbm 0.1 "LocalIM"
```

## License

This project is licensed under the terms of the MIT license.