#!/usr/bin/env bash

python main.py --map=8m\
               --experiment_name=base_experiment_8m\
               --reward_reshape\
               --reward_reshape_method=smac_reward\
               --cuda
