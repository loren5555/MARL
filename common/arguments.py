import argparse

"""
Here are the param for the training

"""


def get_common_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default=None, help='the name of the experiment')
    # the environment setting
    parser.add_argument('--difficulty', type=str, default='7', help='the difficulty of the game')
    parser.add_argument('--game_version', type=str, default='latest', help='the version of the game')
    parser.add_argument('--map', type=str, default='3m', help='the map of the game')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--step_mul', type=int, default=8, help='how many steps to make an action')
    parser.add_argument('--replay_dir', type=str, default='', help='absolute path to save the replay')
    # The alternative algorithms are vdn, coma, central_v, qmix, qtran_base,
    # qtran_alt, reinforce, coma+commnet, central_v+commnet, reinforce+commnet，
    # coma+g2anet, central_v+g2anet, reinforce+g2anet, maven,
    # unfinished: coma+ucb1
    parser.add_argument('--alg', type=str, default='coma+g2anet', help='the algorithm to train the agent')
    parser.add_argument('--n_experiment', type=int, default=1, help='the number of repeating experiment ')
    parser.add_argument('--n_steps', type=int, default=20000000, help='total time steps')
    parser.add_argument('--n_train_steps', type=int, default=20001, help='total train_steps')
    parser.add_argument('--n_episodes', type=int, default=16, help='the number of episodes before once training')
    parser.add_argument('--last_action', type=bool, default=True, help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    # implemented optimizer: RMS, AdamW
    parser.add_argument('--optimizer', type=str, default="AdamW", help='optimizer')
    parser.add_argument('--evaluate_cycle', type=int, default=100, help='how often to evaluate the model')
    parser.add_argument('--evaluate_epoch', type=int, default=64, help='number of the epoch to evaluate the agent')
    parser.add_argument('--model_dir', type=str, default='./model', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')
    parser.add_argument('--log_dir', type=str, default='./log', help='log directory of the policy')
    parser.add_argument('--tensorboard_dir', type=str, default='./runs', help='directory to storge tensorboard files')
    parser.add_argument('--load_model', action="store_true", default=False, help='whether to load the pretrained model')
    parser.add_argument('--evaluate', action="store_true", default=False, help='whether to evaluate the model')
    parser.add_argument('--cuda', action="store_true", default=False, help='whether to use the GPU')
    parser.add_argument('--debug_no_eval', action='store_true', default=False, help='bypass eval to accelerate debugging')

    # reward reshape switch
    parser.add_argument('--reward_reshape', action='store_true', default=False, help='calculate modified reward from state changement instead of smac reward')
    # alternative reshape method
    # smac_reward default SMAC reward
    # static_potential_reward 𝑅′ (𝑠,𝑎,𝑠^′ )=𝑅(𝑠,𝑎,𝑠′ )+𝐹(𝑠,𝑠′) Potential-Based Reward Shaping
    # dynamic_potential_reward
    # static_potential_reward_no_step_reward
    # dynamic_potential_reward_no_step_reward

    # dynamic_potential_reward 𝑅(𝑠,𝑎,𝑠^′ )+𝐹(𝑠,𝑡,𝑠′,𝑡′)
    parser.add_argument('--reward_reshape_method', default='smac_reward', help='reward reshape method, neglected when reward_reshape is False')
    # the potential ratio in potential based reward reshaping method
    parser.add_argument('--potential_ratio', type=float, default=0.5, help='the potential ratio in potential based reward reshaping method')
    parser.add_argument('--weight_increase_factor', type=float, default=0.1, help='the increase speed of the weight of enemy health')
    parser.add_argument('--process_weight_factor', type=float, default=1, help='the weight of time punishment')
    args = parser.parse_args()
    return args


# arguments of coma
def get_coma_args(args):
    # network
    args.rnn_hidden_dim = 128
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.8
    args.anneal_epsilon = 0.00016
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'episode'

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # how often to save the model
    args.save_cycle = 2000

    # how often to update the target_net
    args.target_update_cycle = 100

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of vnd、 qmix、 qtran
def get_mixer_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.qmix_hidden_dim = 32
    args.two_hyper_layers = False
    args.hyper_hidden_dim = 64
    args.qtran_hidden_dim = 64
    args.lr = 5e-4

    # epsilon greedy
    args.epsilon = 1
    args.min_epsilon = 0.05
    anneal_steps = 50000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'step'

    # the number of the train steps in one epoch
    args.train_steps = 1

    # experience replay
    args.batch_size = 32
    args.buffer_size = int(5e3)

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # QTRAN lambda
    args.lambda_opt = 1
    args.lambda_nopt = 1

    # prevent gradient explosion
    args.grad_norm_clip = 10

    # MAVEN
    args.noise_dim = 16
    args.lambda_mi = 0.001
    args.lambda_ql = 1
    args.entropy_coefficient = 0.001
    return args


# arguments of central_v
def get_centralv_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00016
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'episode'

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of central_v
def get_reinforce_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'episode'

    # how often to save the model
    args.save_cycle = 5000

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of coma+commnet
def get_commnet_args(args):
    if args.map == '3m':
        args.k = 2
    else:
        args.k = 3
    return args


def get_g2anet_args(args):
    args.attention_dim = 64
    args.hard = True
    return args


# arguments of ucb1
def get_ucb1_args(args):
    args.attention_dim = 32
    args.ucb1_soft = False  # ucbnet中是否包含softattention
    return args
