#!/usr/bin/env bash

python main.py --map=2s3z\
               --experiment_name=base_experiment_2s3z\
               --reward_reshape\
               --reward_reshape_method=smac_reward\
               --cuda
