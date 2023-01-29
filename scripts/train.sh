#!/usr/bin/env bash

python main.py \
    --map=8m \
    --n_steps=2000000 \
    --n_episodes=4 \
    --alg=coma+g2anet \
    --cuda \
    && shutdown
