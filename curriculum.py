import random

from pommerman.agents import BaseAgent
from pommerman.constants import Action

import action_prune

class SmartRandomAgent(BaseAgent):
    """ random with filtered actions"""
    def act(self, obs, action_space):
        valid_actions = action_prune.get_filtered_actions(obs, [None, None])
        if len(valid_actions) == 0:
            valid_actions.append(Action.Stop.value)
        return random.choice(valid_actions)

    def episode_end(self, reward):
        pass


class SmartRandomAgentNoBomb(BaseAgent):
    """ random with filtered actions"""
    def act(self, obs, action_space):
        valid_actions = action_prune.get_filtered_actions(obs, [None, None])
        if Action.Bomb.value in valid_actions:
            valid_actions.remove(Action.Bomb.value)
        if len(valid_actions) == 0:
            valid_actions.append(Action.Stop.value)
        return random.choice(valid_actions)

    def episode_end(self, reward):
        pass