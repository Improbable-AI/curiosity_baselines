# 6.484 Project spring 2022 @ MIT

Forked from [curiosity baselines](https://github.com/Improbable-AI/curiosity_baselineas) in order to evaluate the performance of self-organising feature maps as curiosity/exploration bonus backbone.

Project by Odin Aleksander Severinsen, Hjalmar Jacob Vinje and Marius Lindegaard for class 6.484 sensorimotor learning spring 2022 at MIT taught by Prof. Pulkit Agrawal.


# Forked README:

## Overview

This is a collection of curiosity algorithms implemented in pytorch on top of the [rlpyt](https://github.com/astooke/rlpyt) deep rl codebase. 

## To-do
1) Add remaining curiosity models
2) Update models directory with more environments

### Available Learning Algorithms
**Policy Gradient** A2C, PPO

**Replay Buffers** (supporting both DQN + QPG) non-sequence and sequence (for recurrent) replay, n-step returns, uniform or prioritized replay, full-observation or frame-based buffer (e.g. for Atari, stores only unique frames to save memory, reconstructs multi-frame observations).

**Deep Q-Learning** DQN + variants: Double, Dueling, Categorical (up to Rainbow minus Noisy Nets), Recurrent (R2D2-style)

**Q-Function Policy Gradient** DDPG, TD3, SAC

### Available Curiosity Algorithms
**Prediction error** ICM, Disagreement

**Count-based** RND

**Learning progress** NDIGO

### Available Environments
* Standard gym environments (mujoco, etc.)
* Atari environments
* SuperMarioBros
* Deepmind PyColab

### Usage

1.  Clone this repo.

2.  If you plan on using mujoco, place your license key "mjkey.txt" in the base directory. This file will be copied in when you start docker
using the Makefile command.

3.  Make sure you have docker installed to run the [image](https://hub.docker.com/repository/docker/echen9898/curiosity_baselines). We recommend
running the GPU image which will work even if you are only using CPUs (labeled version_gpu), but a CPU only image is provided as well.

4.  Edit global.json to customize any volume mount points, port forwarding, and docker image versions from the registry. Information from this file
is read into the Makefile.

5.  The makefile contains some basic commands (we use node to read in information from global.json at the top - it's not used for anything else).
```
make start_docker # start the docker container and drop you in a shell
make start_docker_gpu # start the docker container if running on a machine with GPUs
make stop_docker # stop the docker container and clean up
make clean # clean all subdirectories of pycache files etc.
```

6.  Before running anything, make sure you create an empty directory titled "results" in the base directory.

7.  Run the launch file from the command line, substituting in your preferences for the correct arguments (see rlpyt/utils/launching/arguments.py for a complete list).
```
python3 launch.py -env breakout -alg ppo -curiosity_alg icm -lstm
```

8.  This will launch your experiment in a tmux session titled "experiment". This session will have 3 windows - a window where your code is running, an htop monitoring process, and a window that serves tensorboard to port 12345 (or the port specified in global.json). 

9.  Results folders will be automatically generated in the results directory created in step 6.

10.  Example runs can be found in the models directory. Model weights and exact hyperparameters can be found there for tested environments.

## Notes

For more information on the rlpyt core codebase, please see this [white paper on Arxiv](https://arxiv.org/abs/1909.01500).  If you use this repository in your work or otherwise wish to cite it, please make reference to the white paper.

### Code Organization

The class types perform the following roles:

* **Runner** - Connects the `sampler`, `agent`, and `algorithm`; manages the training loop and logging of diagnostics.
  * **Sampler** - Manages `agent` / `environment` interaction to collect training data, can initialize parallel workers.
    * **Collector** - Steps `environments` (and maybe operates `agent`) and records samples, attached to `sampler`.
      * **Environment** - The task to be learned.
        * **Observation Space/Action Space** - Interface specifications from `environment` to `agent`.
      * **TrajectoryInfo** - Diagnostics logged on a per-trajectory basis.
  * **Agent** - Chooses control action to the `environment` in `sampler`; trained by the `algorithm`.  Interface to `model`.
    * **Model** - Torch neural network module, attached to the `agent`.
    * **Curiosity Model** - Torch neural network module, attached to the `model` which is attached to the `agent`.
    * **Distribution** - Samples actions for stochastic `agents` and defines related formulas for use in loss function, attached to the `agent`.
  * **Algorithm** - Uses gathered samples to train the `agent` (e.g. defines a loss function and performs gradient descent).
    * **Optimizer** - Training update rule (e.g. Adam), attached to the `algorithm`.
    * **OptimizationInfo** - Diagnostics logged on a per-training batch basis.

### Sources and Acknowledgements

This codebase is currently funded by [Amazon MLRA](https://www.amazon.science/research-awards) - we thank them for their support.

Parts of the following open source codebases were used to make this codebase possible. Thanks to all of them for their amazing work!

* [rlpyt](https://github.com/astooke/rlpyt)
* [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros)
* [pycolab](https://github.com/deepmind/pycolab)
* [stable-baselines](https://github.com/hill-a/stable-baselines)

Thanks to [Prof. Pulkit Agrawal](https://people.csail.mit.edu/pulkitag/) and the members of the [Improbable AI lab](https://people.csail.mit.edu/pulkitag/) at MIT CSAIL for their continued guidance and support.





