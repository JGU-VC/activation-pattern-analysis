# Studying the Evolution of Neural Activation Patterns during Training of Feed-Forward ReLU Networks
This repository is the code for the paper: **Studying the Evolution of Neural Activation Patterns during Training of Feed-Forward ReLU Networks**



Setup
-----
1. Make sure you have Pytorch (1.7.1) installed. (This code has been tested on Python 3.9.1).
2. Install the requirements:
    ```
    pip install --user -r requirements.txt
    ```
3. Make sure that `visdom` is running on your machine:
    ```
    mkdir logs
    visdom -env_path ./logs
    ```
    Go to http://localhost:8097


Experiments
-----------
In the following we give all steps required to reproduce the experiments shown in the figures of the paper.

**Early-Phase Plots** (Figure 1)
1. Run `./exp.sh --gpu 0 -d run measures cifar10 early`
  - You can observe the progress using `./exp.sh --stats measures cifar10 early`
  - Multiple GPUs can be used in parallel, just start multiple processes using `./exp.sh --gpu 1 ...`
3. Extract data from log files using: `./paper_util/measures.sh`
3. Alternatively, obtain the data directly from visdom (go to http://localhost:8097)



**Entropycurve-Plot** (Figure 2)
1. Run `./exp.sh --gpu 0 -d run entropycurve cifar10`
  - You can observe the progress using `./exp.sh --stats entropycurve cf10`
  - Multiple GPUs can be used in parallel, just start multiple processes using `./exp.sh --gpu 1 ...`
2. Extract data from log files using: `./plot.py entropycurve ./logs/entropycurve-*`
3. Alternatively, obtain the data directly from visdom (go to http://localhost:8097)


**Full Training-Time Method Plots** (Figure 3)
1. Run `./exp.sh --gpu 0 -d run measures-methods cifar10 full save`
  - This will precompute the network weights required for the last measure (Jaccard Index of the current training step and the final network state).
  - You can observe the progress using `./exp.sh --stats measures-methods cifar10 full save`
  - Multiple GPUs can be used in parallel, just start multiple processes using `./exp.sh --gpu 1 ...`
2. Run `./exp.sh --gpu 0 -d run measures-methods cifar10 full actswithend`
  - This will precompute the network weights required for the last measure (Jaccard Index of the current training step and the final network state).
  - You can observe the progress using `./exp.sh --stats measures-methods cifar10 full save`
  - Multiple GPUs can be used in parallel, just start multiple processes using `./exp.sh --gpu 1 ...`
3. Extract data from log files using: `./paper_util/methods.sh`
4. Alternatively, obtain the data directly from visdom (go to http://localhost:8097)



