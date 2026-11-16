import os
import numpy as np
from datetime import datetime
import sys

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    ppo_runner.alg.num_quantiles = args.num_quantiles
    ppo_runner.alg.taus = torch.linspace(1/(2*args.num_quantiles), 1 - 1/(2*args.num_quantiles), args.num_quantiles, device=ppo_runner.alg.device)

if __name__ == '__main__':
    args = get_args()
    train(args)
