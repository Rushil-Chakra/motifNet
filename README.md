# motifNet
This is a project to impement the network, task structure, and fixed point
analysis of [Flexible multitask computation in recurrent networks utilizes shared dynamical motifs](https://www.nature.com/articles/s41593-024-01668-6).
The original code supporting the paper can be found [here](https://github.com/lauradriscoll/flexible_multitask).
FixedPointFinder original code can be found [here](https://github.com/mattgolub/fixed-point-finder).
## Overview
Right now, I am using `main.py` to run training, find fixed points, save the model, etc.
 The runner script makes use of [Hydra](https://github.com/facebookresearch/hydra) to set task parameters, network and taining parameters, and set the correct launch parameters depending if training is happening on the local machine or a slurm node. Justfile to simplify training is a WIP. If CUDA is available on the running machine, `pytorch` will make use of it.

## Project Organization

```
├── config             <- Composable config tree to work with Hydra.
│   │
│   ├── experiments                 <- Configs for task parameters
│   │
│   └── hydra                       <- Configs for hydra: launcher, logger, etc.
│
├── models             <- Trained models are stored in models/runs/YYYY-MM-DD/HH-MM-SS/
│
├── pyproject.toml     <- Project configuration 
│
├── reports            <- Generated analysis
│   │
│   └── figures                     <- Generated graphics and figures to be used in reporting
│
├── main.py             <- Current working script 
│
└── src                <- Source code for use in this project.
    │
    └── motifnet       <- Makes motif_net a Python module
        │
        ├── __init__.py             <- Makes motifnet a Python module
        │
        ├── FixedPointFinder        <- Edited version of FixedPointFinder to working
        │                              with this codebase. Main difference is switching
        │                              the time and batch axes. 
        │
        └── future                  <- Features that are WIP
```

--------

# Installation
It is highly recommended to first create a virtual environment before installing. 

`uv pip install git+https://github.com/Rushil-Chakra/motifNet`

