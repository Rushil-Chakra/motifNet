from network import MotifNetwork
from train import train
from tasks import TASK_VECTOR_DEFINITION
import task_init

import argparse
import logging


def submit_job():
    return


# TODO: Set up so that i have a commandline entry to run an experiment either with command line args or a json for paramters.
# Should be able to also submit to slurm if that is possible

# set up submission args, experiment args, and model args.
