#!/usr/bin/env bash

python main.py --map=2s3z\
               --experiment_name=dynamic_potential_reward_no_step_reward_2s3z\
               --reward_reshape\
               --reward_reshape_method=dynamic_potential_reward_no_step_reward\
               --cuda\
               --evaluate_cycle=100\
               --n_episodes=16\
               --evaluate_epoch=32\
               --alg=coma+g2anet\
