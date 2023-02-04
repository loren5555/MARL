#!/usr/bin/env bash

python main.py \
    --map=8m \
    --n_steps=2000000 \
    --n_episodes=4 \
    --alg=coma+ucb1 \
    --cuda \
    && shutdown
