#!/usr/bin/env bash

python main.py \
    --map=2s3z \
    --n_steps=2000000 \
    --n_episodes=4 \
    --alg=coma+g2anet \
    --cuda \
    --reward_reshape\
    --reward_reshape_method=static_potential_reward\
    ; shutdown
