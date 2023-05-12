#!/usr/bin/env bash

python main.py --map=2s_vs_1sc\
               --experiment_name=dynamic_potential_reward_2s_vs_1sc\
               --reward_reshape\
               --reward_reshape_method=dynamic_potential_reward\
               --cuda
