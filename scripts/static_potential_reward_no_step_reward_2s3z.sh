#!/usr/bin/env bash

python main.py --map=2s3z\
               --experiment_name=static_potential_reward_no_step_reward_2s3z\
               --reward_reshape\
               --reward_reshape_method=static_potential_reward_no_step_reward\
               --cuda