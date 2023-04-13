import logging

import numpy as np
from smac.env import StarCraft2Env


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

    def reset(self):
        super().reset()

        if self.args.reward_reshape is True:
            # redefine win and death reward
            total_enemy_health = sum([(enemy.health_max + enemy.shield_max) for enemy in self.enemies.values()])
            self.reward_win = 2 * total_enemy_health
            self.reward_death_value = total_enemy_health // self.n_enemies
            self.max_reward = total_enemy_health + self.reward_death_value * self.n_enemies + self.reward_win
            self.reward_defeat = - 0.5 * self.reward_win

            self.state_dict = self.get_state_dict()
            self.last_state_dict = self.state_dict

            # reshape method init
            reset_method_name = "_" + self.args.reward_reshape_method + "_reset"
            getattr(self, reset_method_name)()

    def smac_reward(self):
        # ori smac reward
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

        if self.reward_sparse:
            return 0
        return reward

    def _smac_reward_reset(self):
        pass

    # static potential reward
    def static_potential_reward(self):
        battle_reward = self.smac_reward()
        self.state_dict = self.get_state_dict()

        potential = self._calculate_static_potential()
        F = self.args.gamma * potential - self.last_potential
        reward = battle_reward + F * self.potential_scale

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
            ally_shield = np.zeros(self.n_agents)
        ally_survive = self.n_agents - self.death_tracker_ally.sum()

        enemy_health = self.state_dict["enemies"][:, self.enemy_state_attr_names.index("health")]
        try:
            enemy_shield = self.state_dict["enemies"][:, self.enemy_state_attr_names.index("shield")]
        except ValueError:
            enemy_shield = np.zeros(self.n_enemies)
        enemy_survive = self.n_enemies - self.death_tracker_enemy.sum()

        potential = sum([
            ally_health.sum(),
            ally_shield.sum(),
            ally_survive,
            - enemy_health.sum(),
            - enemy_shield.sum(),
            - enemy_survive
        ])
        return potential

    # static potential reward, without step reward
    def static_potential_reward_no_step_reward(self):
        reward = self.static_potential_reward()
        return reward

    def _static_potential_reward_no_step_reward_reset(self):
        self.reward_sparse = True
        self.max_reward = 1
        self._static_potential_reward_reset()

    # dynamic potential reward
    def dynamic_potential_reward(self):
        battle_reward = self.smac_reward()
        self.state_dict = self.get_state_dict()

        potential = self._calculate_dynamic_potential()
        F = self.args.gamma * potential - self.last_potential
        reward = battle_reward + F * self.potential_scale

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
            ally_shield = np.zeros(self.n_agents)
        ally_survive = self.n_agents - self.death_tracker_ally.sum()

        enemy_health = self.state_dict["enemies"][:, self.enemy_state_attr_names.index("health")]
        try:
            enemy_shield = self.state_dict["enemies"][:, self.enemy_state_attr_names.index("shield")]
        except ValueError:
            enemy_shield = np.zeros(self.n_enemies)
        enemy_survive = self.n_enemies - self.death_tracker_enemy.sum()

        battle_process = self._episode_steps / self.episode_limit
        process_weight = 1 - battle_process
        reverse_process_weight = battle_process + 1

        potential = sum([
            ally_health.sum() * process_weight,
            ally_shield.sum() * process_weight,
            ally_survive * process_weight,
            - enemy_health.sum() * reverse_process_weight,
            - enemy_shield.sum() * reverse_process_weight,
            - enemy_survive * reverse_process_weight,
            - battle_process
        ])
        return potential
