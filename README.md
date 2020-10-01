# Code for "Unifying Non-stationary and Online Clustering of Bandits"

This repository contains implementation of the proposed algorithm DyClu, and baseline algorithms for comparison:
- LinUCB
- dLinUCB and adTS
- CLUB
- oracleLinUCB

For experiments on the synthetic dataset, directly run:
```console
python ClusteredNonStationaryEnvSimulation.py --T 2500 --SMIN 400 --SMAX 2500 --n 30 --m 5 --sigma 0.09
```
To experiment with different environment settings, specify parameters:
- T: number of iterations to run
- SMIN: minimum length of each stationary period
- SMAX: maximum length of each stationary period
- n: number of users
- m: number of unique parameters shared by users
- sigma: standard deviation of Gaussian noise in observed reward

Detailed description of how the simulation environment works can be found in Section 4.1 of the paper.

Experiment results can be found in "./SimulationResults/" folder, which contains:
- "[namelabel]\_[startTime].png": plot of accumulated regret over iteration for each algorithm
- "[namelabel]\_AccRegret\_[startTime].csv": regret at each iteration for each algorithm
- "[namelabel]\_ParameterEstimation\_[startTime].csv": l2 norm between estimated and ground-truth parameter at each iteration for each algorithm
- "Config\_[startTime].json": stores hyper parameters of all algorithms for this experiment

For experiments on LastFM dataset, 
- First download data from https://grouplens.org/datasets/hetrec-2011/
- Then process the dataset following instructions in Section 4.1 of the paper, and put resulting feature vector file and event file under ./Dataset folder
- Run experiments using the provided data loader file "LastFMExperimentRunner.py"
