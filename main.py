import os
from common.marl_logger import MARLLogger

from runner import Runner
from smac.env import StarCraft2Env
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, \
    get_commnet_args, get_g2anet_args, get_ucb1_args

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 解决某个错误 该错误可能由多个conda环境冲突引起

# TODO 服务器训练需要log功能而不能在终端输出
# TODO 添加Tensorboard支持 便于在服务器观看训练状况
if __name__ == '__main__':
    common_args = get_common_args()
    logger = MARLLogger(logger_name="MARL", propagate=False, args=common_args)
    for i in range(8):
        args = common_args
        if args.alg.find('coma') > -1:
            args = get_coma_args(args)
        elif args.alg.find('central_v') > -1:
            args = get_centralv_args(args)
        elif args.alg.find('reinforce') > -1:
            args = get_reinforce_args(args)
        else:
            args = get_mixer_args(args)

        if args.alg.find('commnet') > -1:
            args = get_commnet_args(args)
        if args.alg.find('g2anet') > -1:
            args = get_g2anet_args(args)
        if args.alg.find('ucb1') > -1:
            args = get_ucb1_args(args)

        env = StarCraft2Env(
            map_name=args.map,
            step_mul=args.step_mul,
            difficulty=args.difficulty,
            game_version=args.game_version,
            replay_dir=args.replay_dir,
            window_size_x=1024,
            window_size_y=768
        )

        env_info = env.get_env_info()
        args.n_actions = env_info["n_actions"]
        args.n_agents = env_info["n_agents"]
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]
        args.episode_limit = env_info["episode_limit"]
        logger.starting_log(i, args)

        runner = Runner(env, args)

        if not args.evaluate:
            runner.run(i)
        else:
            win_rate, _ = runner.evaluate()
            print('The win rate of {} is  {}'.format(args.alg, win_rate))
            break
        env.close()
