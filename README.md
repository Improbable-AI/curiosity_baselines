
## Overview

This is a collection of curiosity algorithms implemented in pytorch on top of the [rlpyt](https://github.com/astooke/rlpyt) deep rl codebase. 

### Available Algorithms
**Policy Gradient** A2C, PPO

**Replay Buffers** (supporting both DQN + QPG) non-sequence and sequence (for recurrent) replay, n-step returns, uniform or prioritized replay, full-observation or frame-based buffer (e.g. for Atari, stores only unique frames to save memory, reconstructs multi-frame observations).

**Deep Q-Learning** DQN + variants: Double, Dueling, Categorical (up to Rainbow minus Noisy Nets), Recurrent (R2D2-style)

**Q-Function Policy Gradient** DDPG, TD3, SAC

**Curiosity** ICM, Disagreement, NDIGO

### Available Environments
* Standard gym environments (mujoco, etc.)
* Atari environments
* SuperMarioBros
* Deepmind PyColab

### Usage

1.  Clone this repo.

2.  If you plan on using mujoco, place mjkey.txt in the same directory as the Makefile.

3.  Make sure you have docker installed to run the [image](https://hub.docker.com/repository/docker/echen9898/curiosity_baselines).

4.  The makefile contains some basic commands (we use node to read in the global configuration in global.json, its not used for anything else).
```
make start # start the docker container and drop you in a shell
make stop # stop the docker container
make clean # clean all subdirectories of pycache files etc.
make view # check results in tensorboard
```

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
    * **Distribution** - Samples actions for stochastic `agents` and defines related formulas for use in loss function, attached to the `agent`.
  * **Algorithm** - Uses gathered samples to train the `agent` (e.g. defines a loss function and performs gradient descent).
    * **Optimizer** - Training update rule (e.g. Adam), attached to the `algorithm`.
    * **OptimizationInfo** - Diagnostics logged on a per-training batch basis.
  * **Curiosity Algorithm** - Generates an intrinsic reward signal that can be passed directly through the sampler or in batches to the Algorithm.
    * **Optimizer** - Training update rule (e.g. Adam), attached to the `algorithm`.
    * **OptimizationInfo** - Diagnostics logged on a per-training batch basis.

### Sources and Acknowledgements

Parts of the following open source codebases were used to make this codebase possible. Thanks to all of them for their amazing work!

* [rlpyt](https://github.com/astooke/rlpyt)
* [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros)
* [pycolab](https://github.com/deepmind/pycolab)
* [stable-baselines](https://github.com/hill-a/stable-baselines)

Thanks to Prof. Pulkit Agrawal and the members of the Improbable AI lab at MIT CSAIL for their continued guidance and support.





