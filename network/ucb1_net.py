import torch
import torch.nn as nn
import torch.nn.functional as f
# import numpy as np


# 输入所有agent的obs，输出所有agent的动作概率分布
class UCB1Net(nn.Module):
    def __init__(self, input_shape, args):
        super(UCB1Net, self).__init__()

        # Encoding
        self.encoding = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.h = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        # attention
        if args.ucb1_soft:
            raise NotImplementedError
            # self.q = nn.Linear(args.rnn_hidden_dim, args.attention_dim, bias=False)
            # self.k = nn.Linear(args.rnn_hidden_dim, args.attention_dim, bias=False)
        self.v = nn.Linear(args.rnn_hidden_dim, args.attention_dim)

        self.decoding = nn.Linear(args.rnn_hidden_dim + args.rnn_hidden_dim, args.n_actions)
        self.args = args
        self.input_shape = input_shape

    def ucb1(self, h, size):
        # 用于替换Hard attention
        # input self.h
        # output bool 与某个智能体的通信权重决定的是否通讯
        # TODO 1：仅替代HardAttention， 2：替代G2ANet
        # TODO 重写UCB代码

        # ucb1

        # (n_agents, batch_size, 1, n_agents)
        ucb_weights = torch.ones((self.args.n_agents, size // self.args.n_agents, 1, self.args.n_agents))

        weight_mask = (1 - torch.eye(self.args.n_agents))  # 用于屏蔽ucb对智能体自己的通信评价
        if self.args.cuda:
            ucb_weights = ucb_weights.cuda()
            weight_mask = weight_mask.cuda()

        weight_mask = weight_mask.view(self.args.n_agents, 1, 1, self.args.n_agents)
        masked_weights = ucb_weights * weight_mask

        return f.relu(masked_weights)

    def forward(self, obs, hidden_state):
        size = obs.shape[0]  # batch_size * n_agents
        obs_encoding = f.relu(self.encoding(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)

        h_out = self.h(obs_encoding, h_in)  # 智能体隐藏状态， 用于记忆历史信息
        # v = f.relu(self.v(h_out)).reshape(-1, self.args.n_agents, self.args.attention_dim)  # 注意力隐藏状态编码

        # UCB1
        ucb_weights = self.ucb1(h_out, size)   # (n_agents, batch_size, 1, n_agents)
        ucb_weights = f.softmax(ucb_weights, dim=-1).permute(0, 1, 3, 2)

        # region soft_attention not implemented
        # if self.args.ucb1_soft:
        #     pass
        # else:
        #     pass
        # endregion

        x = (h_out.view(-1, self.args.n_agents, self.args.rnn_hidden_dim) * ucb_weights).sum(dim=-2).reshape(-1, self.args.rnn_hidden_dim)

        final_input = torch.cat([h_out, x], dim=-1)
        output = self.decoding(final_input)

        return output, h_out

    # backup
    # def forward(self, obs, hidden_state):
    #     size = obs.shape[0]  # batch_size * n_agents
    #     obs_encoding = f.relu(self.encoding(obs))
    #     h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
    #
    #     h_out = self.h(obs_encoding, h_in)  # 智能体隐藏状态， 用于记忆历史信息
    #     v = f.relu(self.v(h_out)).reshape(-1, self.args.n_agents, self.args.attention_dim)  # 注意力隐藏状态编码
    #
    #     # UCB1
    #     ucb_weights = self.ucb1(h_out, size)   # (n_agents, batch_size, 1, n_agents)
    #     ucb_weights = f.softmax(ucb_weights, dim=-1).permute(0, 1, 3, 2)
    #
    #     # region soft_attention not implemented
    #     # if self.args.ucb1_soft:
    #     #     pass
    #     # else:
    #     #     pass
    #     # endregion
    #
    #     x = (v * ucb_weights).sum(dim=-2).reshape(-1, self.args.attention_dim)
    #
    #     final_input = torch.cat([h_out, x], dim=-1)
    #     output = self.decoding(final_input)
    #
    #     return output, h_out
