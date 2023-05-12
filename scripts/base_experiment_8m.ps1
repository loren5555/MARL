#神舟z8, win10训练用脚本

conda activate MARL
python main.py --map=8m `
               --experiment_name=base_experiment_8m `
               --reward_reshape `
               --reward_reshape_method=smac_reward `
               --cuda
