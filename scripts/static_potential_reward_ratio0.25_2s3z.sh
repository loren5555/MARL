#!/usr/bin/env bash

python main.py --map=2s3z\
               --experiment_name=static_potential_reward_2s3z_ratio0.25\
               --reward_reshape\
               --reward_reshape_method=static_potential_reward\
               --cuda\
               --potential_ratio=0.25
