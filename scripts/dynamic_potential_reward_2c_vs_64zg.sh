#!/usr/bin/env bash

python main.py --map=2c_vs_64zg\
               --experiment_name=dynamic_potential_reward_2c_vs_64zg\
               --reward_reshape\
               --reward_reshape_method=dynamic_potential_reward\
               --cuda
