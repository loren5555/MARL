#神舟z8, win10训练用脚本

conda activate MARL
python main.py --map=2s3z `
               --experiment_name=static_potential_reward_2s3z `
               --reward_reshape `
               --reward_reshape_method=static_potential_reward `
               --cuda
