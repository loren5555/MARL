import logging

import numpy as np
from smac.env import StarCraft2Env
from common.arguments import get_reward_shaping_args


class RewardShapedStarCraft2Env(StarCraft2Env):
    """
    Modify the reward_battle function in StarCraft2Env.
    When the argument "reward_reshape" in args is False, or the reward reshaping method is smac reward,
    this class is the same as StarCraft2Env.
    """
    def __init__(self, args, **attrs):
        super().__init__(**attrs)
        self.args = args
        self.logger = logging.getLogger("MARL")
        if args.reward_reshape is True:
            self.state_dict = None
            self.last_state_dict = None
            # TODO 通过重写reward_battle方法实现的reward shaping不能修改胜负带来的reward。
            #  若想修改该项，需要重写step函数。没啥卵用，还很复杂，可能再也不do了。
            self.reward_battle = getattr(self, self.args.reward_reshape_method)
            get_reward_shaping_args(args)

    def reset(self):
        super().reset()

        if self.args.reward_reshape is True:
            # redefine win and death reward
            self.reward_win = sum([(enemy.health_max + enemy.shield_max) for index, enemy in self.enemies.items()])
            self.reward_death_value = self.reward_win // self.n_enemies
            self.max_reward = self.reward_win * 2 + self.reward_death_value * self.n_enemies
            self.reward_defeat = - 0.5 * self.reward_win

            self.state_dict = self.get_state_dict()
            self.last_state_dict = self.state_dict

            # reshape method init
            reset_method_name = "_" + self.args.reward_reshape_method + "_reset"
            getattr(self, reset_method_name)()

    def smac_reward(self):
        # ori smac reward
        if self.reward_sparse:
            return 0

        reward = 0
        delta_deaths = 0
        delta_ally = 0
        delta_enemy = 0

        neg_scale = self.reward_negative_scale

        # update deaths
        for al_id, al_unit in self.agents.items():
            if not self.death_tracker_ally[al_id]:
                # did not die so far
                prev_health = (
                        self.previous_ally_units[al_id].health
                        + self.previous_ally_units[al_id].shield
                )
                if al_unit.health == 0:
                    # just died
                    self.death_tracker_ally[al_id] = 1
                    if not self.reward_only_positive:
                        delta_deaths -= self.reward_death_value * neg_scale
                    delta_ally += prev_health * neg_scale
                else:
                    # still alive
                    delta_ally += neg_scale * (
                            prev_health - al_unit.health - al_unit.shield
                    )

        for e_id, e_unit in self.enemies.items():
            if not self.death_tracker_enemy[e_id]:
                prev_health = (
                        self.previous_enemy_units[e_id].health
                        + self.previous_enemy_units[e_id].shield
                )
                if e_unit.health == 0:
                    self.death_tracker_enemy[e_id] = 1
                    delta_deaths += self.reward_death_value
                    delta_enemy += prev_health
                else:
                    delta_enemy += prev_health - e_unit.health - e_unit.shield

        if self.reward_only_positive:
            reward = abs(delta_enemy + delta_deaths)  # shield regeneration
        else:
            reward = delta_enemy + delta_deaths - delta_ally

        return reward

    def _smac_reward_reset(self):
        pass

    def static_potential_reward(self):
        reward = self.smac_reward()
        self.state_dict = self.get_state_dict()

        potential = self._calculate_static_potential()
        F = self.args.gamma * potential - self.last_potential
        reward = reward + F * self.potential_scale

        self.last_potential = potential
        self.last_state_dict = self.state_dict

        return reward

    def _static_potential_reward_reset(self):
        max_potential = self.n_agents * 3 + self.n_enemies * 3
        self.potential_scale = self.args.potential_ratio / (max_potential / self.max_reward)
        self.last_potential = self._calculate_static_potential()

    def _calculate_static_potential(self):
        # calculate static potential from state
        ally_health = self.state_dict["allies"][:, self.ally_state_attr_names.index("health")]
        try:
            ally_shield = self.state_dict["allies"][:, self.ally_state_attr_names.index("shield")]
        except ValueError:
            ally_shield = np.ndarray([0 * self.n_agents])
        ally_survive = self.n_agents - self.death_tracker_ally.sum()

        enemy_health = self.state_dict["enemies"][:, self.enemy_state_attr_names.index("health")]
        try:
            enemy_shield = self.state_dict["enemies"][:, self.enemy_state_attr_names.index("shield")]
        except ValueError:
            enemy_shield = np.ndarray([0 * self.n_enemies])
        enemy_lost_health = 1 - enemy_health
        enemy_lost_shield = 1 - enemy_shield
        enemy_killed = self.death_tracker_ally.sum()

        potential = sum([
            ally_health.sum(),
            ally_shield.sum(),
            ally_survive,
            enemy_lost_health.sum(),
            enemy_lost_shield.sum(),
            enemy_killed
        ])
        return potential

    def dynamic_potential_reward(self):
        reward = self.smac_reward()
        self.state_dict = self.get_state_dict()

        potential = self._calculate_dynamic_potential()
        F = self.args.gamma * potential - self.last_potential
        reward = reward + F * self.potential_scale

        self.last_potential = potential
        self.last_state_dict = self.state_dict

        return reward

    def _dynamic_potential_reward_reset(self):
        max_potential = self.n_agents * 3 + self.n_enemies * 3
        self.potential_scale = self.args.potential_ratio / (max_potential / self.max_reward)
        self.last_potential = self._calculate_dynamic_potential()

    def _calculate_dynamic_potential(self):
        # calculate dynamic potential from state
        ally_health = self.state_dict["allies"][:, self.ally_state_attr_names.index("health")]
        try:
            ally_shield = self.state_dict["allies"][:, self.ally_state_attr_names.index("shield")]
        except ValueError:
            ally_shield = np.ndarray([0 * self.n_agents])
        ally_survive = self.n_agents - self.death_tracker_ally.sum()

        enemy_health = self.state_dict["enemies"][:, self.enemy_state_attr_names.index("health")]
        try:
            enemy_shield = self.state_dict["enemies"][:, self.enemy_state_attr_names.index("shield")]
        except ValueError:
            enemy_shield = np.ndarray([0 * self.n_enemies])
        enemy_lost_health = 1 - enemy_health
        enemy_lost_shield = 1 - enemy_shield
        enemy_killed = self.death_tracker_ally.sum()

        battle_process = self._episode_steps / self.episode_limit
        process_weight = np.exp(- 2 * battle_process)
        reverse_process_weight = 1 - np.exp(- 2 * battle_process)

        potential = sum([
            ally_health.sum() * np.exp(- 2 * battle_process),
            ally_shield.sum() * np.exp(- 2 * battle_process),
            ally_survive * np.exp(- 2 * battle_process),
            enemy_lost_health.sum() * (1 - np.exp(- 2 * battle_process)),
            enemy_lost_shield.sum() * (1 - np.exp(- 2 * battle_process)),
            enemy_killed * (1 - np.exp(- 2 * battle_process))
        ])
        return potential
