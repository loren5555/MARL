#!/usr/bin/env bash

python main.py --map=2s3z\
               --experiment_name=T0605_3\
               --reward_reshape\
               --reward_reshape_method=dynamic_potential_reward\
               --cuda
