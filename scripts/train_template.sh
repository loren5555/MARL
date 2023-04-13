#!/usr/bin/env bash

python main.py --map=3m\
               --experiment_name=base_experiment\
               --reward_reshape\
               --reward_reshape_method=smac_reward\
               --cuda\
               --evaluate_cycle=100\
               --n_episodes=16\
               --evaluate_epoch=32\
               --alg=coma+g2anet\
