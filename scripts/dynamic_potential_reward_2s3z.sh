#!/usr/bin/env bash

python main.py --map=2s3z\
               --experiment_name=dynamic_potential_reward_2s3z\
               --reward_reshape\
               --reward_reshape_method=dynamic_potential_reward\
               --cuda
