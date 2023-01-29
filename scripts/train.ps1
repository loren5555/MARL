#神舟z8, win10训练用脚本

conda activate MARL
python main.py  `
    --map=8m `
    --n_steps=2000000 `
    --n_episodes=4 `
    --alg=coma+g2anet `
    --cuda