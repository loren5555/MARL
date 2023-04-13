#!/usr/bin/env bash

python main.py --map=8m\
               --experiment_name=static_potential_reward_8m\
               --reward_reshape\
               --reward_reshape_method=static_potential_reward\
               --cuda