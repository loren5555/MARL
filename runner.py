import os
import logging

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from agent.agent import Agents, CommAgents
from common.rollout import RolloutWorker, CommRolloutWorker
from common.replay_buffer import ReplayBuffer


class Runner:
    def __init__(self, env, args):
        self.env = env

        if args.alg.find('commnet') > -1 or args.alg.find('g2anet') > -1 or args.alg.find('ucb1') > -1:  # communication agent
            self.agents = CommAgents(args)
            self.rolloutWorker = CommRolloutWorker(env, self.agents, args)
        else:  # no communication agent
            self.agents = Agents(args)
            self.rolloutWorker = RolloutWorker(env, self.agents, args)
        if not args.evaluate and args.alg.find('coma') == -1 and args.alg.find('central_v') == -1 and args.alg.find('reinforce') == -1:  # these 3 algorithms are on-policy
            self.buffer = ReplayBuffer(args)
        self.args = args
        self.win_rates = []
        self.episode_rewards = []
        self.loss = []

        self.logger = logging.getLogger("MARL")
        tb_path = os.path.join(self.args.tensorboard_dir, self.args.alg, self.args.map, self.args.experiment_name)
        self.tb_writer = SummaryWriter(tb_path)

        # 用来保存plt和pkl
        self.save_path = os.path.join(self.args.result_dir, args.alg, args.map, args.experiment_name)

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, num):
        time_steps, train_steps, evaluate_steps = 0, 0, -1  # 总步数， 训练轮数， 评测轮数
        while train_steps < self.args.n_train_steps:
            # eval环节
            if self.args.debug_no_eval is False:
                if train_steps // self.args.evaluate_cycle > evaluate_steps:
                    win_rate, episode_reward = self.evaluate()
                    if train_steps == 0:
                        self.logger.info(f"max_reward: {self.env.max_reward}")
                    self.logger.info(f"time steps: {time_steps}, train steps: {train_steps}, "
                                     f"evaluate steps: {evaluate_steps + 2}, win rate: {win_rate}, "
                                     f"episode reward: {episode_reward}")
                    self.win_rates.append(win_rate)
                    self.episode_rewards.append(episode_reward)
                    self.plt(num)
                    self.tb_writer.add_scalar(f"run{num}/win_rates", win_rate, train_steps)
                    self.tb_writer.add_scalar(f"run{num}/episode_rewards", episode_reward, train_steps)

                    evaluate_steps += 1

            episodes = []
            # 收集self.args.n_episodes个episodes
            for episode_idx in range(self.args.n_episodes):
                episode, _, _, steps = self.rolloutWorker.generate_episode(episode_idx)
                episodes.append(episode)
                time_steps += steps
                # print(_)

            # episode的每一项都是一个(1, episode_len, n_agents, 具体维度)四维数组，下面要把所有episode的的obs拼在一起
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)

            if self.args.alg.find('coma') > -1 or self.args.alg.find('central_v') > -1 or self.args.alg.find('reinforce') > -1:
                # 此处loss仅用作观察网络收敛情况，并非用作判断模型是否收敛
                loss = self.agents.train(episode_batch, train_steps, self.rolloutWorker.epsilon)
                self.tb_writer.add_scalar(f"run{num}/actor loss", loss[0], time_steps)
                self.tb_writer.add_scalar(f"run{num}/critic loss", loss[1], time_steps)
                train_steps += 1
            else:
                self.buffer.store_episode(episode_batch)
                for train_step in range(self.args.train_steps):
                    mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                    self.agents.train(mini_batch, train_steps)
                    train_steps += 1

        # final evaluate
        win_rate, episode_reward = self.evaluate()
        self.logger.info(f"time steps: {time_steps}, train steps: {train_steps}, "
                         f"evaluate steps: {evaluate_steps}, win rate: {win_rate}, "
                         f"episode reward: {episode_reward}")
        self.win_rates.append(win_rate)
        self.episode_rewards.append(episode_reward)
        self.plt(num)
        self.tb_writer.add_scalar(f"run{num}/win_rates", win_rate, train_steps)
        self.tb_writer.add_scalar(f"run{num}/episode_rewards", episode_reward, train_steps)

    def evaluate(self):
        win_number = 0
        episode_rewards = 0
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, win_tag, _ = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            episode_rewards += episode_reward
            if win_tag:
                win_number += 1
        return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch

    def plt(self, num):
        plt.figure()
        plt.ylim([0, 105])
        plt.cla()
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.win_rates)), self.win_rates)
        plt.xlabel('train_step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('win_rates')

        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('train_step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('episode_rewards')

        plt.savefig(self.save_path + '/plt_{}.png'.format(num), format='png')
        np.save(self.save_path + '/win_rates_{}'.format(num), self.win_rates)
        np.save(self.save_path + '/episode_rewards_{}'.format(num), self.episode_rewards)
        plt.close()
