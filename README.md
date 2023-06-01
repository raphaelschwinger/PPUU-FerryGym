# Prediction and Policy learning Under Uncertainty (PPUU) for Autonomous Ferry Navigation


## Forked from [pytorch-PPUU](https://github.com/Atcold/pytorch-PPUU)


## Abstract

Marine Autonomous Surface Ships (MASS) are gaining interest worldwide with the
potential to reshape mobility and freight transport at sea. Collision avoidance
and path planning are central components of the intelligence of a MASS. Deep
Reinforcement Learning (DRL) techniques often learn these capabilities in a
simulator environment. This thesis investigates how to learn path planning and
collision avoidance solely from observational data mitigating the need for a
simulator for training. We construct a state-action dataset of ship trajectories
from recorded Automatic Identification System (AIS) messages. Using this data,
we applied a state-of-the art model-predictive policy learning algorithm. This
includes training an action-conditional forward model and learning a policy
network by unrolling future states and backpropagating errors from a
self-defined cost function. To evaluate our policy, we created an OpenAI gym
environment into which marine traffic data can be loaded. This environment can
also be used to evaluate other algorithms.

## Overview

![Experiments overview](./docs/experiments-overview.png)


## Installation

* clone repo
* start devcontainer or start docker-container with:
    * `docker-compose -f .devcontainer/docker-compose.yml up -d
    * connect editor of choice to container
    * (if you are using vs code dont forget to install python extension)
* preprocess data with `preprocess-dataset.ipynb', '/workspace/data/rev-moenk2' is the default data path (see below)
* download pretrained models and large files with `git lfs pull`

## NN training:
* recipes for different networks are located in `/model_recipes`
* forward model train script: `train_fm.py`
* MPER train script: `train_MPER.py`
* MPUR train script: `train_MPUR.py`


## Evaluation:
* selected pretrained models are located in `/results`
* a `notebook.ipynb` is provided for evaluation of each notebook
* `fm-evaluation.ipynb` for comparing different forward models
* `evaluate_policies.ipynb` for evaluating different policies

## Important PPUU python files:

* `dataloader.py`: for loading data while training and for evaluation
* `models.py`: defines neural network architectures
* `planning.py`: defines policy training algorithms
* `utils.py`: defines helper functions

# Ferry Gym

An [OpenAi Gym](https://www.gymlibrary.ml/) environment to experiment with reinforcement learning in the context of an autonomous ferry.

## Installation

1. Install [pipenv](pipenv.pypa.io)

2. Install dependencies:
Activate the virtual environment:
    
    ```bash
    pipenv shell
    ```

4. Run policy:
        
    ```bash
    python policy-hardcoded.py
    ```

5. Run neural network policies with `run_policy` from `results/utils.py` module

### Important python modules

* `FerryGymEnv/FerryGymEnv.py`: defines the gym environment
* `FerryGymEnv/Ship.py`: defines the agent ship and other vessels
    
# Feature extraction

Our AIS data is stored at a Postgres server at our Institute's server.
To access the data a GraphQL API is provided through [Hasura](https://hasura.io/).

In `preprocessing-dataset.ipynb` you can find a notebook that specifies all necessary steps. By default, the data will be saved in `/data/rev-moenk2/training`. Set a different path in the notebook and create the necessary folders.  You can convert the notebook to a Python script and run it with the following command:
        
        ```bash
        jupyter nbconvert --to script preprocessing-dataset.ipynb
        ```
As the jupyter notebook requires asynchronous calling of the API and the standard python script not, you need to change the following lines in the script:
        
        ```python
        # from
        result = await client.execute_async(query, variable_values=params)
            # ...
        # to
        result = client.execute(query, variable_values=params)
            # ...
        ```

In `FerryGymEnv/load-trajectories` you can find some debugging and visualization code.


## Neural Network policies

* `result/utils.py` provides functions to use a trained policy in the environment

