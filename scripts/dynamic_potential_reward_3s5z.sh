#!/usr/bin/env bash

python main.py --map=3s5z\
               --experiment_name=dynamic_potential_reward_3s5z\
               --reward_reshape\
               --reward_reshape_method=dynamic_potential_reward\
               --cuda

