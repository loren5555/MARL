#!/usr/bin/env bash

python main.py --map=5m_vs_6m\
               --experiment_name=dynamic_potential_reward_5m_vs_6m\
               --reward_reshape\
               --reward_reshape_method=dynamic_potential_reward\
               --cuda
