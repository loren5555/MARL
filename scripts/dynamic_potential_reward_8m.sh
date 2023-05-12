#!/usr/bin/env bash

python main.py --map=8m\
               --experiment_name=dynamic_potential_reward_8m\
               --reward_reshape\
               --reward_reshape_method=dynamic_potential_reward\
               --cuda